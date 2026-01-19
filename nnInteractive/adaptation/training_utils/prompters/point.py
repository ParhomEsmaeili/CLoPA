# #Here we place the abstracted functions which just perform the implementation according to the input only.

# #These funcs should be able to generate empty lists in instances where the sampling region has no valid positions.

# #NOTE: This is not the same as there being no voxels for correction from the start! 
import torch

def uniform_random(binary_mask: torch.Tensor, args: dict):
    '''
    Function which generates n_max spatial coordinate from the input binary mask, assumed to be CBHW(D). Uniform
    application of this heuristic is done on all samples and classes in the batch simultaneously. (I.e., we treat 
    all identically configured prompts across classes and samples in the batch the same way, and so we can vectorise)

    Returns a batchwise separated dicts. One containing lists of prompts for each sample in the batch, the other containing the
    corresponding labels for each prompt in the same order.

    Each list of length n_max (if possible) with tensors denoting spatial coords with shape 1 x n_spatial_dims. If 
    n_max > num possible (which is > 0) then returns a list of num_possible. 
    If num_possible = 0. Then it returns an empty list for that given sample/class combination. 

    Num possible here being the number of voxels in the binary mask which are valid for sampling, for each sample/class combination..

    args: A dict containing the neessary arguments for this heuristic to be applied on a sample/class combination!
    '''
    #Extract n_max from args. It should be the only argument in args for this function. We use a dictionary in order to enable flexibility
    #for abstracting the process of extracting heuristic functions when constructing the prompt generation object.
    
    assert binary_mask.ndim == 5 #Assuming CBHWD input binary mask.

    required_args = {"n_max"} 
    if (set(args.keys()) & required_args) - (set(args.keys()) | required_args):
        raise KeyError(f"Disparity in the required args and provided args for uniform random point sampling function. Do not overload \n"
                       "the function with additional parameters.")
    n_max = args.get("n_max")
    device = binary_mask.device
    #Generate tensor of spatial coords: is Ncoords x N_dim
    
    possible_coords = torch.argwhere(binary_mask.to(device=device, dtype=torch.int32))
    #NOTE: We use int32 here because it is more than enough to represent the array coordinates in the range of the input binary mask.
    if possible_coords.shape[0] <= 0:
        raise ValueError("The provided binary mask has zero valid positions to sample from. Hence, no points can be generated. \n" \
        "should have been handled beforehand.")
    
    #This generates all of the possible coordinates across the batch AND class dimensions 
    #First we get the c_b combinations to filter the possible coords accordingly.
    c_b_combinations = torch.cartesian_prod(torch.arange(binary_mask.shape[0]), torch.arange(binary_mask.shape[1])).to(device=device)
    comb_length = c_b_combinations.shape[1]
    all_matches = (possible_coords[:, None, :comb_length] == c_b_combinations[None, :, :]).all(dim=2)

    #matches is a N_total_possible x N_combinations bool tensor where True indicates that 
    #the corresponding first N (comb_length) values of that coord matches the values from the
    # c_b combination at that index. E.g., False, False, False, True indicates that the coord
    # at that index corresponds to the 4th c_b combination. 

    #Now lets use this to split the possible coords into their respective combinations.
    possible_coords_split_num = all_matches.sum(dim=0) #We sum across the possible_coords dim,
    #as the combinations are unique, so each coord can only belong to one combination. 
    
    # max_count = int(possible_coords_split_num.max().item())
    # #We will use this to split the possible coords into a padded tensor split by combination.
    # if max_count <= 0:
    #     raise ValueError("The provided binary mask has zero valid positions to sample from. Hence, no points can be generated. \n")
    
    #For now lets just use a for loop. We can optimise later if needed.
    output_p = {k: [] for k in range(binary_mask.shape[1])}
    output_plb = {k: [] for k in range(binary_mask.shape[1])}
    for i in range(c_b_combinations.shape[0]):
        if possible_coords_split_num[i] >= n_max:

            #If there are sufficient voxels, return N
            idxs = torch.sort(torch.randint(0, possible_coords_split_num[i] - n_max + 1,(n_max,), device=device)).values + torch.arange(0, n_max, device=device)
            #Yes, this line is a bit overloaded.. we didn't want to use randperm because of time complexity. We sample indices
            #with possible repetition and then offset them to ensure uniqueness. The upper limit is set to ensure the indices stay
            # within bounds after the offset.
            
            coords = possible_coords[torch.nonzero(all_matches[:,i], as_tuple=True)[0]][idxs, comb_length:].clone().to(dtype=torch.int32) 
            #We can use int32 because it would be more than enough to represent the coordinates in the range of the input binary mask.
            
            #NOTE: Don't think the garbage collector is actually doing anything here, so will be commenting it out as it is slowing down
            torch.cuda.empty_cache()
            #Indexing by batch
            output_p[int(c_b_combinations[i, 1].to('cpu'))].extend(list(coords.split(1, 0)))
            output_plb[int(c_b_combinations[i, 1].to('cpu'))].extend([torch.tensor([c_b_combinations[i, 0]], device=device, dtype=torch.int8)] * coords.shape[0])
        
        elif possible_coords_split_num[i] < n_max and possible_coords_split_num[i] != 0:
            #If there are not sufficient voxels greater than the upper limit provided, return the max quantity which is all of them.
            coords = possible_coords[torch.nonzero(all_matches[:,i], as_tuple=True)[0]][:, comb_length:].to(dtype=torch.int32)
            #NOTE: Don't think the garbage collector is actually doing anything here, so will be commenting it out as it is slowing down
            torch.cuda.empty_cache()
            output_p[int(c_b_combinations[i, 1].to('cpu'))].extend(list(coords.split(1, 0)))
            output_plb[int(c_b_combinations[i, 1].to('cpu'))].extend([torch.tensor([c_b_combinations[i, 0]], device=device, dtype=torch.int8)] * coords.shape[0])
        
        elif possible_coords_split_num[i] == 0:
            #NOTE: Don't think the garbage collector is actually doing anything here, so will be commenting it out as it is slowing down
            # del possible_coords
            # gc.collect() 
            torch.cuda.empty_cache()
            output_p[int(c_b_combinations[i, 1].to('cpu'))].extend([])
            output_plb[int(c_b_combinations[i, 1].to('cpu'))].extend([])
        else:
            raise RuntimeError("An unexpected error occurred during point sampling from the provided binary mask.")
    
    return output_p, output_plb