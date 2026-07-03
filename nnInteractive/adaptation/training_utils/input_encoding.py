from functools import lru_cache
from nnInteractive.adaptation.training_utils.general_utils import make_factory 
# from nnInteractive.interaction.point import PointInteraction_stub
import torch 
from scipy.ndimage import distance_transform_edt
from batchgeneratorsv2.helpers.scalar_type import sample_scalar, RandomScalar
from skimage.morphology import disk, ball
import numpy as np

@lru_cache(maxsize=5)
def build_point(radii, use_distance_transform, binarize):
    max_radius = max(radii)
    ndim = len(radii)

    # Create a spherical (or circular) structuring element with max_radius
    if ndim == 2:
        structuring_element = disk(max_radius)
    elif ndim == 3:
        structuring_element = ball(max_radius)
    else:
        raise ValueError("Unsupported number of dimensions. Only 2D and 3D are supported.")

    # Convert the structuring element to a tensor
    structuring_element = torch.from_numpy(structuring_element.astype(np.float32))

    # Create the target shape based on the sampled radii
    target_shape = [round(2 * r + 1) for r in radii]

    if any([i != j for i, j in zip(target_shape, structuring_element.shape)]):
        structuring_element_resized = torch.nn.functional.interpolate(
            structuring_element.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions for interpolation
            size=target_shape,
            mode='trilinear' if ndim == 3 else 'bilinear',
            align_corners=False
        )[0, 0]  # Remove batch and channel dimensions after interpolation
    else:
        structuring_element_resized = structuring_element

    if use_distance_transform:
        # Convert the structuring element to a binary mask for distance transform computation
        binary_structuring_element = (structuring_element_resized >= 0.5).numpy()

        # Compute the Euclidean distance transform of the binary structuring element
        structuring_element_resized = distance_transform_edt(binary_structuring_element)

        # Normalize the distance transform to have values between 0 and 1
        structuring_element_resized /= structuring_element_resized.max()
        structuring_element_resized = torch.from_numpy(structuring_element_resized)

    if binarize and not use_distance_transform:
        # Normalize the resized structuring element to binary (values near 1 are treated as the point region)
        structuring_element_resized = (structuring_element_resized >= 0.5).float()
    return structuring_element_resized

class nnInteractiveUNetEncoding:
    def __init__(self,
        input_handling_configs: dict): 
        

        self.point_conf = input_handling_configs.get('points', None)
        self.box_conf = input_handling_configs.get('bbox', None)
        self.scribble_conf = input_handling_configs.get('scribbles', None)
        self.lasso_conf = input_handling_configs.get('lasso', None)
        self.sequentiality_conf = input_handling_configs.get('sequentiality', None) #sequentiality conf refers to how the sequentiality is handled
        
        

        #For now we will choose to not support box, scribble or lasso to expedite process of finishign this implementation.
        if self.box_conf != None or self.lasso_conf != None:
            raise NotImplementedError('At present, nnInteractiveUNetEncoding only supports point interactions. Box, scribble and lasso interactions are not yet supported.')
        if self.scribble_conf == None:
            raise ValueError('Scribble interaction configuration must be provided for nnInteractiveUNetEncoding.')
        #We will ignore this, its just a byproduct of having a configuration left over from the pretrained model.

        if self.point_conf == None:
            raise ValueError('Point interaction configuration must be provided for nnInteractiveUNetEncoding.')
        if self.point_conf.get('point_interaction_radius') == None:
            raise ValueError('Point radius must be specified in point interaction configuration for nnInteractiveUNetEncoding.')
        if self.point_conf.get('point_interaction_use_etd') == None:
            raise ValueError('Whether to use distance transform must be specified in point interaction configuration for nnInteractiveUNetEncoding.')
        
        if self.scribble_conf.get('preferred_scribble_thickness') == None:
            raise ValueError('Preferred scribble thickness must be specified in scribble interaction configuration for nnInteractiveUNetEncoding.')
        
        if self.sequentiality_conf == None:
            raise ValueError('Sequentiality configuration must be provided for nnInteractiveUNetEncoding.')
        if self.sequentiality_conf.get('interaction_decay') == None:
            raise ValueError('Decay factor must be specified in sequentiality configuration for nnInteractiveUNetEncoding.')
        if self.sequentiality_conf.get('interaction_decay') < 0.0 or self.sequentiality_conf.get('interaction_decay') > 1.0:
            raise ValueError('Decay factor must be between 0.0 and 1.0 in sequentiality configuration for nnInteractiveUNetEncoding.')
        
        self.interaction_channel_dict = {
            'points': {
                1:-4, #1, 0 here refers to the config label value, 1 = fg, 0 = bg
                0: -3
            },
            'bboxes': {
                1: -6,
                0: -5
            },
            'lassos': { 
                1: -6,
                0: -5
            },
            'scribbles': {
                1: -2,
                0: -1
            }
        }

        self.num_interaction_channels = 3 * 2 # 3 unique channels (points, scribbles, boxes/lassos) each with fg and bg

        self.prompt_placer_registry = {
            'points': self.place_point,
            # 'bboxes': self.place_box,
            # 'lassos': self.place_lasso,
            # 'scribbles': self.place_scribble,
        }
    def place_point(self,
                    position: torch.Tensor,
                    interaction_map: torch.Tensor,
                    initialise: bool,
                    binarize: bool = False) -> torch.Tensor:
        """
        Places a point on the interaction map around the specified position.

        Parameters:
        position (torch.Tensor): The unsqueezed (x, y, z) coordinates where the point should be placed.
        shape = 1 x 3
        interaction_map (torch.Tensor): A tensor representing the interaction map where the point
                                        should be placed. The shape should match the volume dimensions.
        binarize (bool): If True, inserts a binary mask. If False, may insert smooth values based on distance.

        Returns:
        torch.Tensor: Updated interaction map with the point added.
        """
        position = tuple(position.squeeze(axis=0).tolist())
        ndim = interaction_map.ndim

        # Determine the radius for each dimension
        radius = tuple([sample_scalar(self.point_conf.get('point_interaction_radius'), d, interaction_map.shape) for d in range(ndim)])

        strel = build_point(radius, self.point_conf.get('point_interaction_use_etd'), binarize)

        # Calculate slice range in each dimension, ensuring it is within the bounds of the interaction map
        bbox = [[position[i] - strel.shape[i] // 2, position[i] + strel.shape[i] // 2 + strel.shape[i] % 2] for i in range(ndim)]
        # detect if bbox is completely outside interaction_map
        if any([i[1] < 0 for i in bbox]) or any([i[0] > s for i, s in zip(bbox, interaction_map.shape)]):
            print('Point is outside the interaction map! Ignoring')
            print(f'Position: {position}')
            print(f'Interaction map shape: {interaction_map.shape}')
            print(f'Point bbox would have been {bbox}')
            return interaction_map
        slices = tuple(slice(max(0, bbox[i][0]), min(interaction_map.shape[i], bbox[i][1])) for i in range(ndim))

        # Calculate where the resized structuring element should be placed within the slices
        structuring_slices = tuple([slice(max(0, -bbox[i][0]), slices[i].stop - slices[i].start + max(0, -bbox[i][0])) for i in range(ndim)])

        # Place the resized structuring element into the interaction map
        torch.maximum(interaction_map[slices], strel[structuring_slices].to(interaction_map.device), out=interaction_map[slices])
        return interaction_map
    
    def __call__(self, prompts_dict, prompts_lbs_dict, init_bool, interaction_channels):
        
        assert interaction_channels.shape[0] == len(prompts_dict) #Asserting batch size matches
        assert interaction_channels.shape[1] == self.num_interaction_channels
        #Slow implementation for now, we just want it up and running.
        for b_idx in range(interaction_channels.shape[0]):
            for ptype, prompts in prompts_dict[b_idx].items():
                if prompts is None:
                    continue
                else:
                    assert ptype in self.prompt_placer_registry, f'Prompt type {ptype} not recognised in prompt placer registry!'
                    assert (torch.isin(torch.stack(prompts_lbs_dict[b_idx][f'{ptype}_labels']).unique().cpu(), torch.Tensor([0,1]))).all(), f'Only binary labels (0 and 1) are supported for prompt type {ptype} in nnInteractiveUNetEncoding!'
                    if any([(torch.stack(prompts_lbs_dict[b_idx][f'{ptype}_labels']) == i).sum() > 1 for i in [0,1]]):
                        raise NotImplementedError('At present, nnInteractiveUNetEncoding only supports single prompts per class (fg/bg). Multiple prompts per class are not yet')

                    # Decay this prompt type's channels once per call (matching inference pattern)
                    if not init_bool:
                        for label in [0, 1]:
                            channel_idx = self.interaction_channel_dict[ptype][label]
                            interaction_channels[b_idx, channel_idx] *= self.sequentiality_conf.get('interaction_decay')

                    for p_idx, p in enumerate(prompts_dict[b_idx][ptype]):
                        label = int(prompts_lbs_dict[b_idx][f'{ptype}_labels'][p_idx].cpu())
                        channel_idx = self.interaction_channel_dict[ptype][label]
                        interaction_channels[b_idx, channel_idx] = self.prompt_placer_registry[ptype](
                            position=p,
                            interaction_map=interaction_channels[b_idx, channel_idx],
                            initialise=init_bool
                        ) 
        return interaction_channels

input_encoding_registry = {
    'nnInteractiveUNetEncoding': make_factory(nnInteractiveUNetEncoding),
}