#This is a file containing various data augmentation functions which may inherit from existing MONAI augmentations to align with the nnu-net dataloading
#stack. Currently, this is empty because no augmentations need to be modified for a single image-channel implementation. 

from monai.transforms import (
    RandRotated,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandZoomd,
    RandZoom,
    Zoom,
    ScaleIntensity,
)
from typing import Any
from monai.utils import convert_data_type
from monai.config.type_definitions import NdarrayOrTensor
from collections.abc import Hashable, Mapping, Sequence
from monai.transforms.transform import MapTransform, RandomizableTransform, LazyTransform
from monai.config.type_definitions import KeysCollection, DtypeLike, SequenceStr
from monai.utils import ensure_tuple_rep
from monai.data.meta_tensor import get_track_meta
from monai.transforms.utils import InterpolateMode, NumpyPadMode
from monai.utils import convert_to_tensor
import numpy as np
import torch

from monai.transforms import RandomizableTransform, RandAffined
import numpy as np

class RandConditionalScaling(RandomizableTransform, LazyTransform):
    def __init__(
        self,
        keys,
        prob=0.3,
        async_prob=0.6,
        scale_range_iso=(0.9, 1.1),
        scale_range_aniso=((0.9, 1.1), (0.9, 1.1), (0.9, 1.1)),
        mode="bilinear",
        padding_mode="border",
        lazy=False
    ):
        super().__init__(prob)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.async_prob = async_prob
        self.iso = RandAffined(
            keys=keys,
            prob=1.0,
            scale_range=scale_range_iso,
            mode=mode,
            padding_mode=padding_mode,
            lazy=lazy
        )

        self.aniso = RandAffined(
            keys=keys,
            prob=1.0,
            scale_range=scale_range_aniso,
            mode=mode,
            padding_mode=padding_mode,
            lazy=lazy
        )

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.iso.lazy = val
        self.aniso.lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandAffined:
        self.iso.set_random_state(seed, state)
        self.aniso.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self
    
    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        self.use_aniso = self.R.random() < self.async_prob

    def __call__(self, data, lazy: bool | None = None):
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        self.randomize()

        if not self._do_transform:
            return data

       
        lazy_ = self.lazy if lazy is None else lazy
       
        if self.use_aniso:
            return self.aniso(data, lazy_)
        else:
            return self.iso(data, lazy_)
        


class RandScaleIntensityClampedd(RandomizableTransform, MapTransform):
    """
    Randomly scale the intensity of input image by ``v = v * (1 + factor)`` where the `factor`
    is randomly picked, and clamp to the original min/max intensity values.
    """

    backend = ScaleIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        factors: tuple[float, float] | float,
        prob: float = 0.1,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            factors: factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            prob: probability of scale.
            channel_wise: if True, scale on each channel separately. Please ensure
                that the first dimension represents the channel of the image if True.
            dtype: output data type, if None, same as input image. defaults to float32.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        if isinstance(factors, (int, float)):
            self.factors = (min(-factors, factors), max(-factors, factors))
        elif len(factors) != 2:
            raise ValueError(f"factors should be a number or pair of numbers, got {factors}.")
        else:
            self.factors = (min(factors), max(factors))
        self.factor = self.factors[0]
        self.channel_wise = channel_wise
        self.dtype = dtype
        self.keys = keys

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> "RandScaleIntensityClampedd":
        super().set_random_state(seed, state)
        return self
    
    def randomize(self, data: Any | None = None) -> None:
            super().randomize(None)
            if not self._do_transform:
                return None 
            if self.channel_wise:
                self.factor = [self.R.uniform(low=self.factors[0], high=self.factors[1]) for _ in range(data.shape[0])]
            else:
                self.factor = self.R.uniform(low=self.factors[0], high=self.factors[1])

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d 
        
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d
    
        assert self._do_transform is True, "self._do_transform must be True here."
        for key in self.key_iterator(d):
            img = d[key]
            if self.channel_wise:
                out = []
                for i, c in enumerate(img):
                    min = torch.min(c)
                    max = torch.max(c) 
                    #We do not use minv and max v here because it will conflict with scaling. we use it AFTER, manually
                    out_channel = ScaleIntensity(minv=None, maxv=None, factor=self.factor[i], dtype=self.dtype)(c)  # type: ignore
                    torch.clamp_(out_channel, min=min, max=max)
                    out.append(out_channel)
                d[key] = torch.stack(out)  # type: ignore
        
            else:
                min = torch.min(img)
                max = torch.max(img)
                ret = ScaleIntensity(minv=None, maxv=None, factor=self.factor, dtype=self.dtype)(img)
                d[key] = torch.clamp_(ret, min=min, max=max)
        return d