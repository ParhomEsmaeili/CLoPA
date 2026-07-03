import logging
from typing import Optional, Sequence, Union
# from SimpleITK import Compose
import nibabel as nib
import numpy as np
# import skimage.measure as measure
import torch
import itk 
import copy
import shutil
import os  
import sys 

from pathlib import Path
from typing import Callable, Sequence
from copy import deepcopy
import tempfile
import warnings
from torch.serialization import DEFAULT_PROTOCOL
from monai.data import Dataset
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import SUPPORTED_PICKLE_MOD, pickle_hashing
from monai.transforms import Compose, Randomizable, RandomizableTrait, Transform, reset_ops_id
from monai.utils import look_up_option
from monai.transforms import Orientation
class WriteImage:
    def __init__(
        self,
        dtypes: dict = None,
        compress:bool=False,
        invert_orient: bool = False,
        monai_reader: bool = True,
        file_ext:str = '.nii.gz'
    ):
        self.dtypes = dtypes 
        self.compress = compress 
        self.invert_orient = invert_orient
        self.monai_reader = monai_reader 
        if not self.monai_reader:
            raise NotImplementedError('Only monai based readers are currently supported for the logic in writing images')
        self.file_ext = file_ext 
    
    def converter(self, image_tensor:torch.Tensor, meta_dict: dict):

        if type(image_tensor) == torch.Tensor:
            #If torch tensor.
            image_metatensor = MetaTensor(image_tensor.clone(), meta=meta_dict)
        else:
            raise TypeError(f'The input must be a torch Tensor.')
        
        return image_metatensor 
    
    def write_itk(
        self, 
        image_np, 
        affine, 
        folder_path:str, 
        dtype: Union[np.dtype, torch.dtype],
        filename_prefix: str = 'image',
        ): 

        if isinstance(image_np, torch.Tensor):
            image_np = image_np.numpy()
        if isinstance(affine, torch.Tensor):
            affine = affine.numpy()
        if len(image_np.shape) >= 2:
            image_np = image_np.transpose().copy() 
            #Transposition operation is necessary to convert between the axis ordering in MONAI IO and default ITK. 
            # MONAI IO functions will be used to read images when adaptation occurs, even if the reader is ITK based.

            # HWD vs DHW axis ordering.
        if dtype:
            image_np = image_np.astype(dtype)

        result_image = itk.image_from_array(image_np)
        
        if affine is not None:
            
            convert_aff_mat = np.diag([-1, -1, 1, 1])
            if affine.shape[0] == 3:
                raise NotImplementedError('We do not yet provide handling for 2D images')
                # if affine.shape[0] == 3:  # Handle RGB (2D Image)
                    # convert_aff_mat = np.diag([-1, -1, 1])

            affine = convert_aff_mat @ affine

            dim = affine.shape[0] - 1
            _origin_key = (slice(-1), -1)
            _m_key = (slice(-1), slice(-1))

            origin = affine[_origin_key]
            spacing = np.linalg.norm(affine[_m_key] @ np.eye(dim), axis=0)
            direction = affine[_m_key] @ np.diag(1 / spacing)


            result_image.SetDirection(itk.matrix_from_array(direction))
            result_image.SetSpacing(spacing)
            result_image.SetOrigin(origin)

        file_path = os.path.join(folder_path, filename_prefix + self.file_ext)
        itk.imwrite(result_image, file_path, self.compress)

        return file_path 
        
    def __call__(self, sample_pair: dict[str, dict], cache_dir:str, sample_name: str):
        
        #We will not permit overwriting existing sample directories, this is an additional measure which prevents 
        #silently having issues with auto-reruns. 
        output_folder = os.path.join(cache_dir, sample_name)
        os.makedirs(output_folder, exist_ok=False)

        im_meta_dict = sample_pair['image']['meta_dict']
        seg_meta_dict = sample_pair['label']['meta_dict']
        
        if not torch.isclose(im_meta_dict['affine'], seg_meta_dict['affine']).all():
            raise Exception("Image and label current affines do not match.")
    

        if self.invert_orient:
            if not torch.isclose(im_meta_dict['original_affine'], seg_meta_dict['original_affine']).all():
                raise Exception("Image and label original affines do not match, cannot invert orientation.")
            if torch.isclose(im_meta_dict['affine'], im_meta_dict['original_affine']).all():
                raise Exception("Image is already in original orientation, cannot invert orientation.") 
            
            #Otherwise, we can proceed. 
            im_metatensor = self.converter(sample_pair['image']['metatensor'], im_meta_dict)
            label_metatensor = self.converter(sample_pair['label']['metatensor'], seg_meta_dict)

            # Undo Orientation on image and label. 
            orig_affine = im_meta_dict.get("original_affine", None)
            if orig_affine is not None:
                orig_axcodes = nib.orientations.aff2axcodes(orig_affine)
                inverse_transform = Orientation(axcodes=orig_axcodes)
                # Apply inverse
                with inverse_transform.trace_transform(False):
                    output_image = inverse_transform(im_metatensor)

                with inverse_transform.trace_transform(False):
                    output_label = inverse_transform(label_metatensor)
            else:
                raise Exception("Failed invert orientation - original_affine is not on the image header")
        else:
            #In this case, we do not invert orientation. We will just leave it in the domain of the input (i.e., 
            #the data distribution passed through for inference!)
            output_image = self.converter(sample_pair['image']['metatensor'], im_meta_dict)
            output_label = self.converter(sample_pair['label']['metatensor'], seg_meta_dict)

        #Assuming that the images are channel first, we will now write out each channel as a separate file (should only
        #be a single channel anyways!) 

        if output_image.shape[0] != 1 or output_label.shape[0] != 1:
            raise NotImplementedError('Writing out multi-channel images is not yet implemented.')
        if output_image.shape[1:] != output_label.shape[1:]:
            raise Exception('Image and label spatial dimensions do not match.') 
    
        im_path = self.write_itk(
            output_image[0].array, 
            output_image.meta['affine'], 
            output_folder, 
            dtype=self.dtypes.get('image'),
            filename_prefix='image') 
        
        label_path = self.write_itk(
            output_label[0].array, 
            output_label.meta["affine"], 
            output_folder, 
            dtype=self.dtypes.get('label'),
            filename_prefix='label') 
        
        return {
            'image': im_path,
            'label': label_path
        }









class ContinualPersistentDataset(Dataset):
    """
    Persistent storage of pre-computed values to efficiently manage larger than memory dictionary format data,
    it can operate transforms for specific fields.  Results from the non-random transform components are computed
    when first used, and stored in the `cache_dir` for rapid retrieval on subsequent uses.

    Adjusted from the original implementation in MONAI to allow for two key changes which are relevant
    for dynamic continual learning scenarios: 1) Ability to amend the cache according to the changes in the
    available data samples. 2) Ability to recompute the cache when the transforms are changed (which they might
    on a continual learning scenario when adaptation is triggered).

    The transforms which are supposed to be cached must implement the `monai.transforms.Transform`
    interface and should not be `Randomizable`. This dataset will cache the outcomes before the first
    `Randomizable` `Transform` within a `Compose` instance.

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
            'image': 'image1.nii.gz',    'image': 'image2.nii.gz',    'image': 'image3.nii.gz',
            'label': 'label1.nii.gz',    'label': 'label2.nii.gz',    'label': 'label3.nii.gz',
            'extra': 123                 'extra': 456                 'extra': 789
        },                           },                           }]

    For a composite transform

        [ LoadImaged(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(96, 96, 96),
                                pos=1, neg=1, num_samples=4, image_key='image', image_threshold=0),
        ToTensord(keys=['image', 'label'])]

    A filename based dataset will be processed by the transform for the
    [LoadImaged, Orientationd, ScaleIntensityRanged] and the resulting tensor written to
    the `cache_dir` before applying the remaining random dependant transforms
    [RandCropByPosNegLabeld, ToTensord] elements for use in the analysis.

    Subsequent uses of a dataset directly read pre-processed results from `cache_dir`
    followed by applying the random dependant parts of transform processing.

        This component will be kept consistent.

        
    During training call `set_data()` to update input data and recompute cache content.

    Note:
        The filenames of the cached files will no longer be the hash keys. Instead, it will be the sample
        names and be placed within the cached sample (image-label) pairs. This is to make it easier
        to update the set of cached samples when the data has the possibility of changing.
         
        
    Lazy Resampling:
        If you make use of the lazy resampling feature of `monai.transforms.Compose`, please refer to
        its documentation to familiarize yourself with the interaction between `PersistentDataset` and
        lazy resampling.

    """

    def __init__(
        self,
        data: dict,
        transform: Sequence[Callable] | Callable,
        cache_dir: Path | str | None,
        cache_subpath: str = "",
        hash_func: Callable[..., bytes] = pickle_hashing,
        pickle_module: str = "pickle",
        pickle_protocol: int = DEFAULT_PROTOCOL,
        reset_ops_id: bool = True,
    ) -> None:
        """
        Args:
            data: input data file paths to load and transform to generate dataset.
                `PersistentDataset` expects input data to be a list of serialisable
                and hashes them as cache keys using `hash_func`.
            transform: transforms to execute operations on input data.
            cache_dir: If specified, this is the location for persistent storage
                of pre-computed transformed data tensors. The cache_dir is computed once, and
                persists on disk until explicitly removed.  Different runs, programs, experiments
                may share a common cache dir provided that the transforms pre-processing is consistent.
                If `cache_dir` doesn't exist, will automatically create it.
                If `cache_dir` is `None`, there is effectively no caching.
            hash_func: a callable to compute hash from data items to be cached.
                defaults to `monai.data.utils.pickle_hashing`.
            pickle_module: string representing the module used for pickling metadata and objects,
                default to `"pickle"`. due to the pickle limitation in multi-processing of Dataloader,
                we can't use `pickle` as arg directly, so here we use a string name instead.
                if want to use other pickle module at runtime, just register like:
                >>> from monai.data import utils
                >>> utils.SUPPORTED_PICKLE_MOD["test"] = other_pickle
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save,
                and ``monai.data.utils.SUPPORTED_PICKLE_MOD``.
            pickle_protocol: can be specified to override the default protocol, default to `2`.
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save.
            reset_ops_id: whether to set `TraceKeys.ID` to ``Tracekys.NONE``, defaults to ``True``.
                When this is enabled, the traced transform instance IDs will be removed from the cached MetaTensors.
                This is useful for skipping the transform instance checks when inverting applied operations
                using the cached content and with re-created transform instances.

        """
        data_names = [n for n in data.keys()] #Assuming data is a dict of sample_name: sample_dict
        data = [deepcopy(d) for d in data.values()]
        self.data_names = data_names 
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, transform=transform)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_subpath = cache_subpath 
        self.hash_func = hash_func
        self.pickle_module = pickle_module
        self.pickle_protocol = pickle_protocol
        self.reset_ops_id = reset_ops_id

    def set_data(self, data: dict, reset_cache: bool = False):
        """
        Set the input data for cache. Delete all the out-dated cache content if resetting cache is 
        enabled.Will require calling on the dataset again in order to re-cache the data.

        If reset_cache is False, then it will just store the full data, and downstream implementation
        should handle the missing cache files (i.e., the new samples).
        """
        data_names = [n for n in data.keys()] #Assuming data is a dict of sample_name: sample_dict
        data = [deepcopy(d) for d in data.values()] #Conversion to a sequence of dicts for the actual
        #dataset input.
        self.data_names = data_names
        self.data = data
        
        if reset_cache:
            #Now we will clear the cached files according to the sample names.
            for sample_name in data_names:
                if os.path.exists(os.path.join(self.cache_dir, sample_name, self.cache_subpath, f'cached_sample.pt')):
                #We will look for the cached file, and delete it. But will not touch the files of the sample
                #on the disk which are not in the stored cache (i.e., the original image-label pair).
                # This is to prevent accidental deletion of data. 
                    os.remove(os.path.join(self.cache_dir, sample_name, self.cache_subpath, f'cached_sample.pt'))
                else:
                    continue
        else:
            pass 

        
    def _pre_transform(self, item_transformed):
        """
        Process the data from original state up to the first random element.

        Args:
            item_transformed: The data to be transformed

        Returns:
            the transformed element up to the first identified
            random transform object

        """
        first_random = self.transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )
        item_transformed = self.transform(item_transformed, end=first_random, threading=True)

        if self.reset_ops_id:
            reset_ops_id(item_transformed)
        return item_transformed

    def _post_transform(self, item_transformed):
        """
        Process the data from before the first random transform to the final state ready for evaluation.

        Args:
            item_transformed: The data to be transformed (already processed up to the first random transform)

        Returns:
            the transformed element through the random transforms

        """
        first_random = self.transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )
        if first_random is not None:
            item_transformed = self.transform(item_transformed, start=first_random)
        return item_transformed

    def _cachecheck(self, sample_dict: dict, sample_name: str):
        """
        A function to cache the expensive input data transform operations
        so that huge data sets (larger than computer memory) can be processed
        on the fly as needed, and intermediate results written to disk for
        future use.

        Args:
            item_transformed: The current data element to be mutated into transformed representation

        Returns:
            The transformed data_element, either from cache, or explicitly computing it.

        Warning:
            The current implementation does not encode transform information as part of the
            hashing mechanism used for generating cache names when `hash_transform` is None.
            If the transforms applied are changed in any way, the objects in the cache dir will be invalid.

        """
        # hashfile = None #This was redundant, we will ALWAYS assume that if the cache dir doesn't exist that it will raise an error!
        if self.cache_dir is not None and os.path.exists(self.cache_dir):
            hash_dir = os.path.join(self.cache_dir, sample_name, self.cache_subpath)
            os.makedirs(hash_dir, exist_ok=True)
            hashfile = os.path.join(hash_dir, f"cached_sample.pt")
        else:
            raise Exception("Cache directory does not exist, cannot proceed with caching mechanism.")
        #First we try to load a cached file and return that.
        if hashfile == None:
            raise Exception("We need a hashfile path, directory was not specified, cannot proceed with caching mechanism.")
        if os.path.exists(hashfile):  # cache hit
            try:
                return torch.load(hashfile, weights_only=False)
            except PermissionError as e:
                if sys.platform != "win32":
                    raise e
            except RuntimeError as e:
                if "Invalid magic number; corrupt file" in str(e):
                    warnings.warn(f"Corrupt cache file detected: {hashfile}. Deleting and recomputing.")
                    os.remove(hashfile)
                else:
                    raise e
        else:
            # If it doesn't exist then we make it.
            _item_transformed = self._pre_transform(deepcopy(sample_dict))  # keep the original hashed
        
            try:
                # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
                #       to make the cache more robust to manual killing of parent process
                #       which may leave partially written cache files in an incomplete state
                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_hash_file = os.path.join(Path(tmpdirname), f"cached_sample.pt")
                    torch.save(
                        obj=_item_transformed,
                        f=temp_hash_file,
                        pickle_module=look_up_option(self.pickle_module, SUPPORTED_PICKLE_MOD),
                        pickle_protocol=self.pickle_protocol,
                    )
                    if os.path.exists(temp_hash_file) and not os.path.exists(hashfile):
                        # On Unix, if target exists and is a file, it will NOT be replaced silently.
                        #We do not want to accidentally overwrite existing cache files, if a file needs to be replaced then
                        #it should have been deleted first.
                        try:
                            shutil.move(str(temp_hash_file), hashfile)
                        except FileExistsError:
                            raise Exception(f"Cache file {hashfile} already exists, cannot overwrite existing cache. If changing"
                                                " transforms, please delete the existing cache files first.")
                    elif not os.path.exists(temp_hash_file):
                        raise Exception(f"Temporary cache file {temp_hash_file} was not created successfully.")
                    elif os.path.exists(temp_hash_file) and os.path.exists(hashfile):
                        raise Exception(f"Cache file {hashfile} already exists, cannot overwrite existing cache. If changing"
                                            " transforms, please delete the existing cache files first.")
                    
            except Exception as e:  # project-monai/monai issue #3613
                raise Exception(f'Error {e} encountered when trying to write cache file. Cannot continue with experiment!')
            return _item_transformed

    def _transform(self, index: int):
        pre_random_item = self._cachecheck(self.data[index], self.data_names[index])
        return self._post_transform(pre_random_item)
    

    





# if __name__ == "__main__":

    # im_path = '/home/parhomesmaeili/MY METHOD/AdaptiveIS/debug_image/lung_001.nii.gz'#BraTS2021_00266.nii.gz'
    # label_path = '/home/parhomesmaeili/MY METHOD/AdaptiveIS/debug_image/lung_001_label.nii.gz'#BraTS2021_00266.nii.gz' #just use the same.
    # input_dict = {
    #     'image': im_path,
    #     'label': label_path
    # }

    # invert_orientation = True 

    # writer = WriteImage(
    #     dtypes={
    #         'image': np.float32,
    #         'label': np.uint8
    #     },
    #     compress=False,
    #     invert_orient=invert_orientation,
    #     monai_reader=True,
    #     file_ext='.nii.gz'
    # )

    # load_transforms_1 = Compose([
    #     LoadImaged(keys=['image', 'label'], reader='ITKReader'),
    #     EnsureChannelFirstd(keys=['image', 'label']),
    #     Orientationd(keys=['image', 'label'], axcodes='RAS')
    # ])
    
    # if not invert_orientation: #In this case the image needs to be saved with the orientation of the loaded domain.
    #     #So we do not apply the orientation transform when re-loading.
    #     load_transforms_2 = Compose([
    #         LoadImaged(keys=['image', 'label'], reader='ITKReader'),
    #         EnsureChannelFirstd(keys=['image', 'label']),
    #     ])
    # else: #In this case, the image needs to be saved back in its original orientation, so we apply the same orientation transform
    # #when re-loading.
    #     load_transforms_2 = Compose([
    #         LoadImaged(keys=['image', 'label'], reader='ITKReader'),
    #         EnsureChannelFirstd(keys=['image', 'label']),
    #         Orientationd(keys=['image', 'label'], axcodes='RAS')
    #     ])

    # initial_loaded_dict = load_transforms_1(input_dict)

    # im_tensor = torch.from_numpy(initial_loaded_dict['image'].array)
    # label_tensor = torch.from_numpy(initial_loaded_dict['label'].array)
    # im_meta_dict = {
    #     'affine': initial_loaded_dict['image'].meta['affine'], 
    #     'original_affine': torch.from_numpy(initial_loaded_dict['image'].meta['original_affine'])
    # }
    # label_meta_dict = {
    #     'affine': initial_loaded_dict['label'].meta['affine'], 
    #     'original_affine': torch.from_numpy(initial_loaded_dict['label'].meta['original_affine'])
    # }

    # reformatted_loaded_dict = {
    #     'image': {
    #         'metatensor': im_tensor,
    #         'meta_dict': im_meta_dict
    #     },
    #     'label': {
    #         'metatensor': label_tensor,
    #         'meta_dict': label_meta_dict
    #     }
    # }

    # saved_paths_1 = writer(
    #     sample_pair=reformatted_loaded_dict, 
    #     cache_dir='/home/parhomesmaeili/MY METHOD/AdaptiveIS/debug_image/cache_dir',
    #     sample_name='test_sample_1')
    
    # #Now we will save the same image, but without the function. instead using a nibabel write directly, to just
    # #see how the image has been changed in the loaded domain. 
    # nib.save(
    #     nib.Nifti1Image(
    #         initial_loaded_dict['image'].array[0], 
    #         None
    #     ),
    #     '/home/parhomesmaeili/MY METHOD/AdaptiveIS/debug_image/cache_dir/test_sample_1/nib_image.nii.gz'
    # )


    # #Now load the saved images again, and see if it matches the original loaded ones. 

    # reloaded_dict = load_transforms_2({
    #     'image': saved_paths_1['image'],
    #     'label': saved_paths_1['label']
    # })

    # if not torch.allclose(
    #     reloaded_dict['image'], 
    #     initial_loaded_dict['image']
    # ):
    #     print('Image arrays do not match after save and reload! Please check.')
    # if not torch.allclose(
    #     reloaded_dict['label'], 
    #     initial_loaded_dict['label']
    # ):
    #     print('Label arrays do not match after save and reload! Please check.'
    # )
   
    # print('pause')