import torch 
from typing import Union
def extractor(x: dict, y: tuple[Union[tuple,str, int]]): 
    '''
    This general purpose function is adapted from a lambda function which iterates through the dict using the tuple, 
    where the order/index denotes the depth. 
    
    Once tuple is empty it stops (also for NoneType it stops, i.e. just returns the current dict.) and returns the
    item from the provided tuple path. 

    Inputs: 
        x - A dictionary which is populated and will be extracted from
        y - A tuple consisting of the iterative path through the nested dict.
    '''
    if not y:
        return x
    else:
        if not isinstance(y, tuple):
            raise TypeError('y - path in dict must be a tuple')
        
        if y: #If y exists and we are iterating through it still:
            if not isinstance(x, dict):
                raise TypeError('x must be a dictionary')
            if x == {}:
                raise ValueError('The input dict must be populated otherwise we cannot extract anything.')
            else:
                return extractor(x[y[0]], y[1:]) 
      

def update_binary_mask_freeform(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Modifies an existing binary mask by setting specified coordinates to zero.
    
    Args:
        coords (torch.Tensor): An (N x D) tensor containing coordinates, where N is the number of coordinates
                              and D is the number of spatial dimensions of the mask.
        mask (torch.Tensor): An existing binary mask.
    
    Returns:
        torch.Tensor: The modified binary mask with zeros at `coords` locations.
    
    Raises:
        ValueError: If the number of dimensions in `coords` does not match the mask's dimensions.
    """
    # Early return if coords is empty
    if coords.numel() == 0:
        return mask
    
    if coords.shape[1] != mask.dim():
        raise ValueError(f"Dimension mismatch: coords has {coords.shape[1]} spatial dimensions, but mask has {mask.dim()} dimensions.")
    
    # Ensure mask and coords are on the same device
    device = mask.device 
    coords = coords.to(device)
    
    # Apply bounds check for each coordinate, dimension by dimension
    valid_mask = torch.all((coords >= 0) & (coords < torch.tensor(mask.shape, device=device)), dim=1)
    
    # Filter valid coordinates
    valid_coords = coords[valid_mask].to(device)  # Move valid_coords to the same device as mask

    if valid_coords.numel() > 0:  # Ensure there are valid coordinates
        # indices = tuple(valid_coords.to(torch.int32).T)  
        # Convert valid coordinates to indices using int32. should be sufficient
        #for the vast majority of voxel counts. unless maybe we started working with images of size 10000 x 10000 x 10000 etc.
        #int32 should be sufficient for indexing, as it can represent indices in the range of the mask dimensions.
        
        #We move the index handling in-line to prevent unnecessary variable assignments.
        mask[tuple(valid_coords.to(torch.int32).T)] = False  # Set valid positions to False
    
    # Handling VRAM. 
    valid_coords = valid_coords.cpu()
    coords = coords.cpu()
    valid_mask = valid_mask.cpu()
    torch.cuda.empty_cache() #Not sure if this is deallocating anything? #TODO diagnose this when we have more time.
    return mask.to(dtype=torch.bool) 
