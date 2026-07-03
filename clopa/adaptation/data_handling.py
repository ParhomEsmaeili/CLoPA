import os
import sys
import copy 
import warnings
from math import ceil 
import shutil
import json 
# import base64
from clopa.adaptation.io_operations import WriteImage, ContinualPersistentDataset
from monai.transforms import (
    ToDeviced,
    EnsureTyped,
    LoadImaged,
    Compose,
    DivisiblePadd,
    SpatialPadd,
    CenterSpatialCropd,
    EnsureChannelFirstd, 
    Orientationd,
    RandCropByPosNegLabeld,
    RandCropByLabelClassesd,
    NormalizeIntensityd,
    SpatialResampled,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandSimulateLowResolutiond,
    RandRotate90d,
    RandScaleIntensityd,
    RandFlipd
)
from clopa.adaptation.training_utils.dataloading_augs import RandConditionalScaling, RandScaleIntensityClampedd
from clopa.adaptation.training_utils.general_utils import make_factory
from torch.utils.data import RandomSampler, Sampler 
import torch
import numpy as np
from monai.data import DataLoader
from collections import defaultdict
import inspect 

class IndexedSampler(Sampler):
    def __init__(self, indices):
        self.indices = list(indices)
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

def find_conflicts(d: dict) -> dict:
    val2keys = defaultdict(list)
    for key, vals in d.items():
        for v in vals:
            val2keys[v].append(key)
    return {v: keys for v, keys in val2keys.items() if len(keys) > 1}

  
class DataHandler:
    '''
    This is a class for handling data saving/loading, and data splitting strategies. In short, it
    manages the data used for adaptation purposes. 

    Initialisation is only for variables which are static across the adaptation process, e.g. the io operations
    or the mechanism for generating dataloaders. (NOT the transforms). 
    '''
    def __init__(
        self,
        dataset_constructor_config: dict, 
        io_operations_config: dict,
        dataloaders_transforms: dict = None,
        ):
        self.dataset_constructor_config = dataset_constructor_config
        self.io_operations_config = io_operations_config

        self.image_writer = WriteImage(
            dtypes = io_operations_config.get('dtypes', {
                'image': np.float32,
                'label': np.uint8
            }),
            compress=io_operations_config.get('compress', False),
            invert_orient=io_operations_config.get('invert_orient', False),
            monai_reader= io_operations_config.get('monai_reader', True),
            file_ext=io_operations_config.get('file_ext', '.nii.gz')
            )

        self.supported_load_transforms = {
            'ToDeviced': make_factory(ToDeviced), #Using the same factory function as the other transform.
            'EnsureTyped': make_factory(EnsureTyped),
            'LoadImaged': make_factory(LoadImaged),
            'EnsureChannelFirstd': make_factory(EnsureChannelFirstd),
            'Orientationd': make_factory(Orientationd),
            'SpatialPadd': make_factory(SpatialPadd),
            'DivisiblePadd': make_factory(DivisiblePadd),
            'CenterSpatialCropd': make_factory(CenterSpatialCropd),
            'RandCropByPosNegLabeld': make_factory(RandCropByPosNegLabeld),
            'RandCropByLabelClassesd': make_factory(RandCropByLabelClassesd), 
            'NormalizeIntensityd': make_factory(NormalizeIntensityd),
            'SpatialResampled': make_factory(SpatialResampled),
            'RandRotated': make_factory(RandRotated),
            'RandConditionalScaling': make_factory(RandConditionalScaling),
            'RandScaleIntensityClampedd': make_factory(RandScaleIntensityClampedd),
            'RandZoomd': make_factory(RandZoomd),
            'RandGaussianNoised': make_factory(RandGaussianNoised),
            'RandGaussianBlurd': make_factory(RandGaussianSmoothd),
            'RandAdjustContrastd': make_factory(RandAdjustContrastd),
            'RandSimulateLowResolutiond': make_factory(RandSimulateLowResolutiond),
            'RandRotate90d': make_factory(RandRotate90d),
            'RandFlipd': make_factory(RandFlipd), #Using the same factory as rotate90 since they have the same args.
            'RandScaleIntensityd': make_factory(RandScaleIntensityd),
            
        } 

        self.supported_dataset_constructors = [
            'ContinualPersistentDataset'
        ]
        
        #Setting the dataset obj and dataloader transforms. Dataset is an actual object which holds samples so 
        #cannot be stored in a checkpoint, whereas the dataloader transforms corresponding to this dataset/cache
        #can be stored. Hence it is not forced to be set to Nonetype. 
        self.datasets = None
        self.dataloaders_transforms = dataloaders_transforms 
    def create_dataset(
        self,
        case_list: list[str],
        memory_buffer_disk: dict[str, dict[str, str]], #This is a nested dict, outer keys are sample names,
        #inner keys are image/label with corresponding filepaths as values.
        deterministic_transforms: dict[str, any],
        dynamic_transforms: dict[str, any],
        cache_dir: str, #The directory for all things being cached within this experiment.
        cache_subpath: str, #The subpath within the cache dir for this dataset's cache.
        check_reset_cache: bool
        ):
        '''
        A function which give a set of configured MONAI transforms, returns a dataset object which can load samples using
        those configured transformations. 
        '''
    

        if any([key not in self.supported_load_transforms.keys() for key in deterministic_transforms.keys()]):
            unsupported_keys = [key for key in deterministic_transforms.keys() if key not in self.supported_load_transforms.keys()]
            raise NotImplementedError(f'The following load transforms are not supported: {unsupported_keys}'
                )
        if any([key not in self.supported_load_transforms.keys() for key in dynamic_transforms.keys()]):
            unsupported_keys = [key for key in dynamic_transforms.keys() if key not in self.supported_load_transforms.keys()]
            raise NotImplementedError(f'The following load transforms are not supported: {unsupported_keys}'
                )
        
        dataset_list = {
            sample_name: memory_buffer_disk.get(sample_name) for sample_name in case_list
        }
    
        if not all([os.path.exists(paths['image']) and os.path.exists(paths['label']) for paths in dataset_list.values()]):
            raise FileNotFoundError('One or more image or label paths in the case list do not exist.')
        
        transforms = [
            self.supported_load_transforms[transf](input_vars) for transf, input_vars in deterministic_transforms.items()
        ]
        transforms.extend([
            self.supported_load_transforms[transf](input_vars) for transf, input_vars in dynamic_transforms.items()
        ])
        #Any non supported variables will just be flagged at initialistion time of the transform itself.
        #The wrapping of the functions/transforms is not compatible with auto-detection of arguments.

        # composed_transforms = Compose(transforms) 
        
        if check_reset_cache:
            #If we are resetting the dataset, we could either use the reset cache operation, or just wipe all of the cache files here. Lets just do it here.
            for case_name in case_list:
                if os.path.exists(os.path.join(cache_dir, case_name, cache_subpath)):
                    shutil.rmtree(os.path.join(cache_dir, case_name, cache_subpath)) 


        if self.dataset_constructor_config.get('dataset_constructor', None) == None:
            raise ValueError('No dataset constructor specified in dataset_constructor_config!')
        elif self.dataset_constructor_config.get('dataset_constructor', None) == 'ContinualPersistentDataset':
            return ContinualPersistentDataset(
                data=dataset_list,
                transform=transforms,#composed_transforms,
                cache_dir=cache_dir,
                cache_subpath=cache_subpath
        )
        else:
            raise NotImplementedError(f'Dataset constructor {self.dataset_constructor_config.get("dataset_constructor")} is not implemented yet.')
    
    def update_dataset(
        self,
        dataset_obj: ContinualPersistentDataset,
        case_list: list[str],
        memory_buffer_disk: dict[str, dict[str, str]], #This is a nested dict, outer keys are sample names,
        #inner keys are image/label with corresponding filepaths as values.
    ):
        '''
        A function which updates the samples in a continual persistent dataset object, assumes the transforms
        are unchanged.
        '''
        if not isinstance(dataset_obj, ContinualPersistentDataset):
            raise TypeError('dataset_obj must be an instance of ContinualPersistentDataset to update samples.')
        
        dataset_dict = {
            sample_name: memory_buffer_disk.get(sample_name) for sample_name in case_list
        }
        dataset_obj.set_data(
            data=dataset_dict,
            reset_cache=False
        )

        return dataset_obj 


    def create_dataloaders(
        self,
        data_split: dict[str, dict[str, list[str]]],
        memory_buffer_disk: dict[str, dict[str, str]], #This is a nested dict, outer keys are sample names,
        #inner keys are image/label with corresponding filepaths as values.
        adaptation_planner: dict,
        memory_buffer_dir: str
    ):
        '''
        This is a function which generates a dataloader from the adaptation planner and available datalist.
        It makes use of the persistent continual dataset class for this purpose.
        '''
    
        dataloaders = dict()
        if self.dataloaders_transforms == None and self.datasets == None:
            #In this case, there is not any precedent loaded, so we will need to create the dataset, set the dataloader
            # etc from scratch. This is the first time training has occured.
            self.dataloaders_transforms = dict()  
            self.datasets = dict()
            #Setting all of these from scratch, not the dataloader however which will be created each callback.
            for split_name, split_dict in data_split.items():
                dataloaders[split_name] = dict() 
                self.datasets[split_name] = dict() 
                self.dataloaders_transforms[split_name] = dict()
                #Now we create a pair of datasets for each split name, train
                #and validation splits. 
            
                #NOTE: First we verify that there is no overlap between the splits.
                if find_conflicts(split_dict):
                    conflicts = find_conflicts(split_dict)
                    raise ValueError(f'Data split conflict detected! The following samples are assigned to multiple splits: {conflicts}')

                for data_type, case_list in split_dict.items():
                    deterministic_transforms = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('deterministic_transforms')
                    dynamic_transforms = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('dynamic_transforms')
                    assert deterministic_transforms != None, f'Deterministic transforms must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    assert dynamic_transforms != None, f'Random transforms must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    assert type(deterministic_transforms) == dict, f'Deterministic transforms must be a dict for {data_type} {split_name} dataloader in adaptation planner!'
                    assert type(dynamic_transforms) == dict, f'Random transforms must be a dict for {data_type} {split_name} dataloader in adaptation planner!'
                    assert deterministic_transforms != {} , f'Deterministic transforms cannot be empty for {data_type} {split_name} dataloader in adaptation planner!'
                    
                    #L
                    
                    #We don't make any assertions about random transforms, they can be empty but must be a dict.
                    self.datasets[split_name][data_type] = self.create_dataset(
                        case_list=case_list,
                        memory_buffer_disk=memory_buffer_disk,
                        deterministic_transforms=deterministic_transforms,
                        dynamic_transforms=dynamic_transforms,
                        cache_dir=memory_buffer_dir,
                        cache_subpath=os.path.join('cache', split_name, data_type),
                        check_reset_cache=True #In this case, just be careful! It shouldn't have any cache yet though. 
                        )
                    #Stashing the transforms used for this dataloader for future reference.
                    self.dataloaders_transforms[split_name][data_type] = {
                        'deterministic_transforms': deterministic_transforms,
                        'dynamic_transforms': dynamic_transforms
                    }
                    #we will now create a dataloader for this split and data type.

                    #Now lets extract the parameters for the dataloader from the adaptation planner. And check
                    #them first. 
                    batch_size = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('batch_size')
                    num_workers = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('num_workers')
                    pin_memory = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('pin_memory')
                    assert batch_size != None, f'Batch size must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    assert num_workers != None, f'Num workers must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    assert pin_memory != None, f'Pin memory must be specified for {data_type} {split_name} dataloader in adaptation planner!'

                    assert type(batch_size) == int and batch_size > 0, f'Batch size must be a positive integer for {data_type} {split_name} dataloader in adaptation planner!'
                    assert type(num_workers) == int and num_workers >= 0, f'Num workers must be a non-negative integer for {data_type} {split_name} dataloader in adaptation planner!'
                    assert type(pin_memory) == bool, f'Pin memory must be a boolean for {data_type} {split_name} dataloader in adaptation planner!'
                    
                    if data_type == 'train':
                        gen = torch.Generator(device="cpu")
                        gen.manual_seed(torch.seed())
                        sampler = RandomSampler(self.datasets[split_name][data_type], generator=gen)
                    elif data_type == 'val':
                        #We will  not use an indexed sampler, we will rely on statistics instead by using a random
                        #sampler with replacement......., we set the iterations per epoch separate to the quantity
                        #of batches generated by dataset size and batch size.
                        
                        # sampler = IndexedSampler(range(len(self.datasets[split_name][data_type])))  
                        gen = torch.Generator(device="cpu")
                        gen.manual_seed(torch.seed())
                        sampler = RandomSampler(self.datasets[split_name][data_type], generator=gen)
                    else:
                        raise ValueError(f'Unknown data type {data_type} specified in data split!')
                    
                    dataloaders[split_name][data_type] = DataLoader(
                            dataset=self.datasets[split_name][data_type],
                            batch_size=batch_size,
                            sampler=sampler, 
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                        )
                    
        elif self.dataloaders_transforms != None and self.datasets == None:
            # it is auto-rerun and so 
            # the dataset obj is not yet initialised. If there was no dataloader there could not have been
            # a dataset object!
            self.datasets = dict()
            for split_name, split_dict in data_split.items():
                self.datasets[split_name] = dict()
                dataloaders[split_name] = dict() 
                #Now we create a pair of dataloaders for each split name, train
                #and validation dataloaders. 

                #NOTE: First we verify that there is no overlap between the splits.
                if find_conflicts(split_dict):
                    conflicts = find_conflicts(split_dict)
                    raise ValueError(f'Data split conflict detected! The following samples are assigned to multiple splits: {conflicts}')

                for data_type, case_list in split_dict.items():
                    #We will check if the transforms have changed since last time. If they have not, then we
                    # can just update the samples and have those cached.
                    deterministic_transforms = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('deterministic_transforms')
                    dynamic_transforms = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('dynamic_transforms')
                    assert deterministic_transforms != None, f'Deterministic transforms must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    assert dynamic_transforms != None, f'dynamic  transforms must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    planner_transforms = {
                        'deterministic_transforms': deterministic_transforms,
                        'dynamic_transforms': dynamic_transforms
                    }
                    #Regardless, we need to create the dataset obj here. Just need to flag whether the cache needs to be checked for deletion.
                    if any(
                        [
                            self.dataloaders_transforms[split_name][data_type][k] != planner_transforms[k]
                            for k in planner_transforms.keys()
                        ]
                        ): 
                    
                        self.datasets[split_name][data_type] = self.create_dataset(
                            case_list=case_list,
                            memory_buffer_disk=memory_buffer_disk,
                            deterministic_transforms=planner_transforms.get('deterministic_transforms'),
                            dynamic_transforms=planner_transforms.get('dynamic_transforms'),
                            cache_dir=memory_buffer_dir,
                            cache_subpath=os.path.join('cache', split_name, data_type),
                            check_reset_cache=True #check for reset since transforms have changed.
                            )
                    else:
                        #In this case, it hasn't changed, so we just recreate the dataset obj with a new set of dataset samples. Don't need to remove
                        #cache.
                        self.datasets[split_name][data_type] = self.create_dataset(                            
                            case_list=case_list,
                            memory_buffer_disk=memory_buffer_disk,
                            deterministic_transforms=planner_transforms.get('deterministic_transforms'),
                            dynamic_transforms=planner_transforms.get('dynamic_transforms'),
                            cache_dir=memory_buffer_dir,
                            cache_subpath=os.path.join('cache', split_name, data_type),
                            check_reset_cache=False #No need to change here as transforms are the same.
                        )
                    #we will now create a dataloader for this split and data type.

                    batch_size = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('batch_size')
                    num_workers = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('num_workers')
                    pin_memory = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('pin_memory')
                    assert batch_size != None, f'Batch size must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    assert num_workers != None, f'Num workers must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    assert pin_memory != None, f'Pin memory must be specified for {data_type} {split_name} dataloader in adaptation planner!'

                    assert type(batch_size) == int and batch_size > 0, f'Batch size must be a positive integer for {data_type} {split_name} dataloader in adaptation planner!'
                    assert type(num_workers) == int and num_workers >= 0, f'Num workers must be a non-negative integer for {data_type} {split_name} dataloader in adaptation planner!'
                    assert type(pin_memory) == bool, f'Pin memory must be a boolean for {data_type} {split_name} dataloader in adaptation planner!'

                    if data_type == 'train':
                        gen = torch.Generator(device="cpu")
                        gen.manual_seed(torch.seed())
                        sampler = RandomSampler(self.datasets[split_name][data_type], generator=gen)
                    elif data_type == 'val':
                        # sampler = IndexedSampler(range(len(self.datasets[split_name][data_type])))  
                        gen = torch.Generator(device="cpu")
                        gen.manual_seed(torch.seed())
                        sampler = RandomSampler(self.datasets[split_name][data_type], generator=gen)
                    else:
                        raise ValueError(f'Unknown data type {data_type} specified in data split!')
                    
                    dataloaders[split_name][data_type] = DataLoader(
                            dataset=self.datasets[split_name][data_type],
                            batch_size=batch_size,
                            sampler=sampler, 
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                        )
                    
        elif self.dataloaders_transforms != None and self.datasets != None:
            #In this case, we are going to continue adaptation with stored obj online.
             
            #Transforms have been set, so we should have cached samples and a dataset. We will verify this is the
            #case. If the transforms have changed, we will need to recreate the dataset, otherwise we can just update 
            # the cache with the new samples in the dataset.
    
            for split_name, split_dict in data_split.items():
                dataloaders[split_name] = dict() 
                #Now we create a pair of dataloaders for each split name, train
                #and validation dataloaders. 


                #NOTE: First we verify that there is no overlap between the splits.
                if find_conflicts(split_dict):
                    conflicts = find_conflicts(split_dict)
                    raise ValueError(f'Data split conflict detected! The following samples are assigned to multiple splits: {conflicts}')


                for data_type, case_list in split_dict.items():
                    #We will check if the transforms have changed since last time. If they have not, then we
                    # can just update the samples and have those cached.
                    deterministic_transforms = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('deterministic_transforms')
                    dynamic_transforms = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('dynamic_transforms')
                    assert deterministic_transforms != None, f'Deterministic transforms must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    assert dynamic_transforms != None, f'dynamic  transforms must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    planner_transforms = {
                        'deterministic_transforms': deterministic_transforms,
                        'dynamic_transforms': dynamic_transforms
                    }

                    if any(
                        [
                            self.dataloaders_transforms[split_name][data_type][k] != planner_transforms[k]
                            for k in planner_transforms.keys()
                        ]
                        ):  
                        self.datasets[split_name][data_type] = self.create_dataset(
                            case_list=case_list,
                            memory_buffer_disk=memory_buffer_disk,
                            deterministic_transforms=planner_transforms.get('deterministic_transforms'),
                            dynamic_transforms=planner_transforms.get('dynamic_transforms'),
                            cache_dir=memory_buffer_dir,
                            cache_subpath=os.path.join('cache', split_name, data_type),
                            check_reset_cache=True #It must reset.
                            )
                    else:
                        #In this case, it hasn't changed, so we just update the dataset samples IF the dataset
                        #object has not been created yet (e.g. on auto-rerun)
                        if self.datasets == None:
                            raise ValueError('Datasets object is None, cannot update samples without a dataset \n'
                                'object existing! Should have been routed through the other branch of this function')
                        self.datasets[split_name][data_type] = self.update_dataset(                            
                            dataset_obj=self.datasets[split_name][data_type],
                            case_list=case_list,
                            memory_buffer_disk=memory_buffer_disk
                        )
                    #we will now create a dataloader for this split and data type.
                    batch_size = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('batch_size')
                    num_workers = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('num_workers')
                    pin_memory = adaptation_planner[f'{data_type}_handlers'][split_name]['dataloading_config'].get('pin_memory')
                    assert batch_size != None, f'Batch size must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    assert num_workers != None, f'Num workers must be specified for {data_type} {split_name} dataloader in adaptation planner!'
                    assert pin_memory != None, f'Pin memory must be specified for {data_type} {split_name} dataloader in adaptation planner!'

                    assert type(batch_size) == int and batch_size > 0, f'Batch size must be a positive integer for {data_type} {split_name} dataloader in adaptation planner!'
                    assert type(num_workers) == int and num_workers >= 0, f'Num workers must be a non-negative integer for {data_type} {split_name} dataloader in adaptation planner!'
                    assert type(pin_memory) == bool, f'Pin memory must be a boolean for {data_type} {split_name} dataloader in adaptation planner!'

                    if data_type == 'train':
                        gen = torch.Generator(device="cpu")
                        gen.manual_seed(torch.seed())

                        sampler = RandomSampler(self.datasets[split_name][data_type], generator=gen)
                    elif data_type == 'val':
                        # sampler = IndexedSampler(range(len(self.datasets[split_name][data_type])))  
                        gen = torch.Generator(device="cpu")
                        gen.manual_seed(torch.seed())

                        sampler = RandomSampler(self.datasets[split_name][data_type], generator=gen)
                    else:
                        raise ValueError(f'Unknown data type {data_type} specified in data split!')
                    
                    dataloaders[split_name][data_type] = DataLoader(
                        dataset=self.datasets[split_name][data_type],
                        batch_size=batch_size,
                        sampler=sampler, 
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                    )
        else:
            raise Exception('Unknown state for dataloader transforms and dataset obj in data handler!')
        return dataloaders 
        
    def write_image(
        self,
        sample_pair: dict,
        memory_buffer_dir: str, 
        sample_name: str
    ):
        '''
        A function which writes an image-label pair to disk using the configured WriteImage class.'''
        return self.image_writer(
            sample_pair,
            memory_buffer_dir,
            sample_name
        )
        
    def data_splitting(
        self,
        data_splitting_config: dict,
        existing_datasplit: dict[str, list[str]],
        new_cases_datalist: list[str],
    ):
        '''
        A function which handles data splitting strategies based on the provided configuration.
        
        This data splitting is configured on the basis of the new samples acquired, not the entire
        dataset which is kept in memory. I.e., it is only appending new samples to existing dataset
        splits. 

        inputs:
        existing_datasplit: dict[str, list[str]]: existing data split dictionary (or empty dict if no prior 
        saved data split exists).
        new_cases_datalist: list[str]: list of new cases to be split according to the data splitting strategy.

        returns:
        split_data: dict: dictionary containing the split data according to the specified strategy,
            e.g., {'train': list of train_cases, 'val': list of val_cases} for holdout strategy,
            or just {'train': list of train_cases} for strategies without validation split,
            or for k-fold cross validation it may return {'fold_0': {'train': list of train_cases_fold_0, 'val': list of val_cases_fold_0}, ... } etc.

        '''
        strategy = data_splitting_config.get('name', None)
         
        if strategy == None:
            raise ValueError('No data splitting strategy specified in data_splitting_config!')
        
        if strategy == 'static_hold_out':
            #No check in place for ensuring that sufficient samples are available, this was handled in the
            #trigger criterion.
            train_fraction = data_splitting_config.get('fraction_train')
            if train_fraction is None or not (0 < train_fraction < 1):
                raise ValueError('fraction_train must be specified in data_splitting_config and be between 0 and 1 for hold_out strategy.')
            # np.random.shuffle(new_cases_datalist)
            
            #NOTE: We removed the shuffle because it could cause auto-rerun inconsistencies on the data split.
            #The shuffling of the dataset is better handled in the dataloader AND in the experiment configured.
            #NOTE: It could have been handled on the application end, but it would be extremely convoluted!  
            #any reordering of the cases should be done on the validation end. 
            split_idx = ceil(len(new_cases_datalist) * train_fraction) #We go with ceiling because we want to give priority to training samples if 
            train_cases = new_cases_datalist[:split_idx]
            val_cases = new_cases_datalist[split_idx:]
            
            new_datasplit = copy.deepcopy(existing_datasplit)
            if new_datasplit == {}:
                new_datasplit['fold_0'] = {}
                new_datasplit['fold_0']['train'] = []
                new_datasplit['fold_0']['val'] = []
            
            new_datasplit['fold_0']['train'].extend(train_cases)
            new_datasplit['fold_0']['val'].extend(val_cases)
            return new_datasplit
        else:
            raise NotImplementedError(f'Data splitting strategy {strategy} is not implemented yet.')                    
         