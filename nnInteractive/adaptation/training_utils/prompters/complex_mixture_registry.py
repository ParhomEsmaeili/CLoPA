# #Registry of prompters, takes input predictions, and target labels and produces prompts. 

# import torch
# from nnInteractive.adaptation.training_utils.prompters.utils import update_binary_mask_freeform, extractor 
# from nnInteractive.adaptation.training_utils.general_utils import make_factory
# from typing import Union
# import copy
# import itertools
# import warnings 
# from monai.data import MetaTensor 
# import re 
# class BaseMixture:
#     def __init__(self, sim_device: torch.device, semantic_id_dict: dict):
#         self.supported_prompts = ['points', 'scribbles', 'bboxes', 'lassos']
#         self.discrete_variables = [i + '_labels' for i in self.supported_prompts] #The labels are discrete variables, 
#         #the spatial coordinates need not necessarily be discrete. To permit sub-voxel coordinates in future implementations.
#         self.sim_device = sim_device
#         self.semantic_id_dict = semantic_id_dict 
    
#     def check_config_availability(self, input_configs: dict[dict], prompter_type: str):
#         '''
#         Function which loops through a set of configurations which are being utilised in order to check that
#         configuration parameters required for configuring the prompt simulation classes are
#         provided. I.e., no missing parameters/loose ends. 

#         Raises errors if there are supported prompts which have no configuration provided. 
#         '''
#         for config_name, config in input_configs.items():
#             for p_type in self.supported_prompts:
#                 if p_type not in config.keys():
#                     raise Exception(f'The prompt type {p_type} was not provided in the configuration dictionary: {config_name} for {prompter_type}') 

#     def shuffle_list(self, input_list: list, sort_criterion: str = None):
#         '''
#         Can be used to sort a list according to a criterion, e.g. random permutation to sort a list of heuristics or toggles.
#         Or other criteria in future implementations, e.g. error-region component size.    
#         ''' 
#         if sort_criterion is None:
#             return input_list 
#         elif sort_criterion.title() == 'Random':
#             #We create a random permutation of integers from 0 to len(sublist) - 1.
#             indices = torch.randperm(len(input_list)).to(int) 
#             return [input_list[i] for i in indices]
#         else:
#             raise NotImplementedError('The list sort criterion provided was not supported')
    
#     def filter_empty(self,
#             prompts:Union[list[torch.Tensor], None], 
#             prompts_lbs: Union[list[torch.Tensor], None] 
#             ):
#         '''
#         Function which performs checks on the prompts and prompts labels to ensure that they match one another, and
#         then evaluates how to filter the values according to what the prompts and prompt labels contain (or lack-thereof)
#         '''
#         if not prompts and not prompts_lbs: #In instances where list is empty, or where it is NoneType both eval as False.
#             prompts = None
#             prompts_lbs = None 
#         elif bool(prompts) ^ bool(prompts_lbs): #Logical XOR, if they do not match then there is an error!
#             raise Exception('One of the prompts and prompt labels evaluated as a False, whereas the other was True. Mismatch.')
#         elif bool(prompts) & bool(prompts_lbs): #If both are true-types that are not empty.
#             if len(prompts) != len(prompts_lbs):
#                 raise Exception('Non-matching quantity of prompts and prompts labels')
#             else:
#                 pass #Made explicit for readability.
        
#         return prompts, prompts_lbs 
            
#     def filter_empty_dict(self, 
#             prompts_dict: dict[str, Union[list[torch.Tensor], None]], 
#             prompts_lbs_dict: dict[str, Union[list[torch.Tensor], None]], 
#             ):
#         '''
#         Function which filters out the empty prompt types in output dicts into NoneTypes.

#         inputs:

#         prompts: A dictionary, separated by prompt types containing the lists of prompt inputs.
#         '''

#         if not {f'{i}_labels' for i in set(prompts_dict)} == set(prompts_lbs_dict):
#             raise Exception('The prompts dict and the prompts labels dict somehow did not contain the same prompt types')
                
#         for prompt_type in prompts_dict.keys():
#             p_list = prompts_dict[prompt_type]
#             p_lbs_list = prompts_lbs_dict[f'{prompt_type}_labels']

#             p_updated, p_lbs_updated = self.filter_empty(p_list, p_lbs_list)
            
#             prompts_dict.update({prompt_type: p_updated})
#             prompts_lbs_dict.update({f'{prompt_type}_labels': p_lbs_updated})

#         return prompts_dict, prompts_lbs_dict
    
#     def discrete_checker(self, data: Union[list[torch.Tensor], None]):
#         '''
#         Function which checks and converts input data which is containing discrete values (e.g. perhaps some prompts, 
#         labels), to ensure that they are torch.int datatypes. Also moves it to the device specified.

#         inputs: 
        
#         data: An optional list of torch tensors (size is not relevant).
#         If NoneType then ignores. 

#         returns: 

#         data: A list of torch tensors but in the correct datatype (torch.int) or NoneType for unused instances.
#         '''

#         if not data: #If empty list, or NoneType, just pass through.
#             pass #Explicit logic provided here.... for debugging.
#         else:
#             for idx, tensor in enumerate(data):
#                 if tensor.is_complex():
#                     raise Exception('No complex numbers should be possible.')
                
#                 #NOTE: Removed the floating point 
#                 elif tensor.is_floating_point():
#                     warnings.warn(f'The input data {tensor} should have been provided as a torch int type, converting to int64.')
#                     data[idx] = tensor.to(dtype=torch.int64) #Just use int64, it is safe as the data is typically small
#                     #anyways. 
#         return data    
    
#     def device_processor(self, data: Union[list[torch.Tensor], None], device: torch.device):
#         '''
#         Function which checks and/or moves it to the device specified.

#         inputs: 
#         data: An optional list of torch tensors (size is not relevant).
#         If NoneType then ignores. 

#         returns: 
#         data: A list of torch tensors but on the correct device/ or NoneType for unused instances.
#         '''
#         if not data: #If empty list, or NoneType, just pass through.
#             pass #Explicit logic provided here.... for debugging.
#         else:
#             for idx, tensor in enumerate(data):
#                 if tensor.device != device:
#                     data[idx] = tensor.to(device=device)
#         return data
        
#     def output_processor(self, prompts_dict: dict, prompts_lbs_dict: dict):
#         '''
#         Function which post-processes the prompts dictionary and prompts labels dictionary to ensure they are in 
#         the correct format, and on the correct device.
#         '''
        
#         #We first run it through a function which filters empty lists, checks for any inconsistencies between 
#         #prompts and labels across all prompt types. 
#         prompts_dict, prompts_lbs_dict = self.filter_empty_dict(prompts_dict=prompts_dict, prompts_lbs_dict=prompts_lbs_dict)

#         #We then perform any datatype processing, and move to the cpu device for passing back through the API.
#         for ptype, p_lb_type in zip(prompts_dict.keys(), prompts_lbs_dict.keys()):
#             tmp_plist = copy.deepcopy(prompts_dict[ptype])
#             tmp_plb_list = copy.deepcopy(prompts_lbs_dict[f'{ptype}_labels'])
#             tmp_plist = self.device_processor(data=tmp_plist, device=torch.device('cpu'))
#             tmp_plb_list = self.device_processor(data=tmp_plb_list, device=torch.device('cpu')) 

#             if ptype in self.discrete_variables:
#                 tmp_plist = self.discrete_checker(data=tmp_plist)
#             if p_lb_type in self.discrete_variables:
#                 # prompts_lbs_dict[f'{ptype}_labels'] = self.discrete_checker(prompt_list=tmp_plb_list, device=torch.device('cpu'))
#                 tmp_plb_list = self.discrete_checker(data=tmp_plb_list)
            
#             prompts_dict[ptype] = tmp_plist 
#             prompts_lbs_dict[p_lb_type] = tmp_plb_list
#         return prompts_dict, prompts_lbs_dict


#     def reformat_im(self, im: Union[dict, None], ptypes: tuple):
#         '''
#         Function which creates a deepcopy of the interaction memory prompt information and places them on the 
#         selected simulation device picked at config. 
        
#         It takes the entire set of interaction states (as any non-desirable memory states should be discarded in the 
#         front-end im-handler). For each prompt type it then merges the prompts across the states (with no further operation).

#         Prompts extracted are provided in the ordering of the interaction memory.

#         Downstream functions may be required for implementing any handling of redundancies (e.g. copies), 
#         or forward propagation of prompts (e.g. bbox) etc.

#         NOTE: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         NOTE: The underlying assumption for this function is that the each instance of a prompt is provided as a torch
#         TENSOR. As such, a set of prompts for a given state must only be a list of torch tensors.
        
#         inputs:

#         im: An optional dictionary (None when empty, e.g. for initialisation) containing the full list of IM which 
#         may be used for prompt generation.

#         The only relevant field here is: 'interaction_torch_format', which contains two subfields:

#         interactions: A dictionary, separated by prompt type, containing the list of torch tensors with the corresponding prompt info.
#         interactions_labels: A dictionary, separated by prompt type, containing the list of torch tensors with the 
#         corresponding class-integer code for the prompts.

#         p_types: A tuple of strings denoting the prompt types. Must match with the keys of each state in the im.

#         returns: 

#         None, None (if no IM) OR ordered_prompts, ordered_prompts_labels (each is a dict split by ptype with a list of torch tensors on device). 
#         '''
#         if not im:
#             return None, None
#         else:
#             im_copy = copy.deepcopy(im)
            
#             ordered_prompts_dict = dict.fromkeys(ptypes, [])
#             ordered_prompts_lb_dict = dict.fromkeys(ptypes, [])

#             if 'Interactive Init' in im_copy.keys():
#                 init_state = im_copy.pop('Interactive Init') 
#                 #We first extract the interactive init dict. 
#                 cartesian_prod = ptypes  #tuple(itertools.product(('Interactive Init',), ptypes))
                
#                 for ptype in cartesian_prod:
#                     p = extractor(init_state, ('interaction_torch_format', 'interactions', ptype))
#                     p_lb = extractor(init_state,('interaction_torch_format', 'interactions_labels', ptype))

                
#                     assert type(p) == list and type(p_lb) == list 
#                     #Put a check for each item in the list
#                     assert all([type(p_item) == torch.Tensor or type(p_item) == MetaTensor for p_item in p])
#                     assert all([type(p_item) == torch.Tensor or type(p_item) == MetaTensor for p_item in p_lb]) 

#                     #We then merge into the ordered prompts dicts.

#                     temp_p = copy.deepcopy(ordered_prompts_dict[ptype])
#                     temp_p.extend([p_item.to(device=self.sim_device) for p_item in p])
                    
#                     temp_plb = copy.deepcopy(ordered_prompts_lb_dict[ptype])
#                     temp_plb.extend([p_item.to(device=self.sim_device) for p_item in p_lb])

#                     ordered_prompts_dict[ptype] = temp_p
#                     ordered_prompts_lb_dict[ptype] = temp_plb 
                
#             #We then extract from the interactive edits, in order. If the remaining state keys are empty, then the 
#             # cartesian product will return an empty tuple so that the iterator will just terminate.

#             #Orders the remaining keys according to the edit iter num:
#             edit_names_list = list(im_copy.keys())
#             edit_names_list.sort(key=lambda test_str : list(map(int, re.findall(r'\d+', test_str))))
            
#             cartesian_prod = tuple(itertools.product(edit_names_list, ptypes))
        
#            #Extracts using all of the tuple combinations between state names and prompt types. 

#             for state_name, ptype in cartesian_prod:
#                 #Extract
#                 p = extractor(im_copy, (state_name, 'interaction_torch_format', 'interactions', ptype))
#                 p_lb = extractor(im_copy, (state_name, 'interaction_torch_format', 'interactions_labels', ptype))
                
#                 assert type(p) == list and type(p_lb) == list 
#                 #Put a check for each item in the list
#                 assert all([type(p_item) == torch.Tensor or type(p_item) == MetaTensor for p_item in p])
#                 assert all([type(p_item) == torch.Tensor or type(p_item) == MetaTensor for p_item in p_lb]) 

#                 #We then merge into the ordered prompts dicts.
#                 temp_p = copy.deepcopy(ordered_prompts_dict[ptype])
#                 temp_p.extend([p_item.to(device=self.sim_device) for p_item in p])
                
#                 temp_plb = copy.deepcopy(ordered_prompts_lb_dict[ptype])
#                 temp_plb.extend([p_item.to(device=self.sim_device) for p_item in p_lb])

#                 ordered_prompts_dict[ptype] = temp_p 
#                 ordered_prompts_lb_dict[ptype] = temp_plb
        
#             for ptype in ptypes:
#                 if len(ordered_prompts_dict[ptype]) != len(ordered_prompts_lb_dict[ptype]):
#                     raise Exception(f'Mismatch between quantity of prompts and prompt labels for {ptype}')
                
#                 if all([i is None for i in ordered_prompts_dict[ptype]]) ^ all([i is None for i in ordered_prompts_lb_dict[ptype]]):
#                     raise Exception[f'Mismatch between whether all prompts and all prompt labels in memory are NoneTypes for {ptype}']
                
#                 elif all([i is None for i in ordered_prompts_dict[ptype]]) ^ all([i is None for i in ordered_prompts_lb_dict[ptype]]):
#                     ordered_prompts_dict[ptype] = None 
#                     ordered_prompts_lb_dict[ptype] = None 

#             del im_copy 

#             return ordered_prompts_dict, ordered_prompts_lb_dict 
        

    
# class BasicValidOnlyMixture(BaseMixture): 
#     def __init__(self, use_mem: bool, **kwargs):
#         self.use_mem = use_mem
#         super().__init__(**kwargs)

#     def rm_intra_prompt_spat_repeats(self, ordered_prompts_dict, ordered_prompts_lb_dict):
#         '''
#         Lazy implementation of removing intra-prompt type repeats for spatial prompts (which should not occur
#         if interaction memory is being used for prompt generation conditioning, which is the only common instance 
#         where this function would be used anyways. 
        
#         Does not make any considerations of putting placeholders etc for retaining the structure of IM.


#         NOTE: Assumption is that all prompts are provided as a list of torch tensors. 
        
#         NOTE: Second assumption, is that this will only implemented on intra-prompt. Hence, checks will not be imp-
#         lemented across prompts. I.e., for verifying that scribble coords do not overlap with point coordinates.
#         The prompt generation strategy needs to handle this! 
        
#         NOTE: Third assumption is that given that this is only for valid prompt placements (according to GT), hence
#         why we are capable of filtering in the chronological order for prompt placement. Otherwise, it would filter
#         in a manner that is not intuitive on a UI (i.e. overwriting!).

#         inputs: 

#         ordered_prompts_dict: A dictionary, split by ptype (typically extracted from interaction memory, 
#         but also can be from the current call of prompt generation) 

#         ordered_prompts_lbs_dict: A dictionary, split by ptype (typically extracted from interaction memory,
#         but also can be from the current call of prompt generation).

#         For instances where it comes from IM, these are ordered with respect to the IM states. Within the IM states
#         (i.e. even within the current call of prompt generation) these are ordered with respect to the order in which
#         they were simulated.

#         NOTE: Assumption, there is no prompt type which is registered as NoneType yet! This will occur on the final pass...?

#         returns:

#         filtered order_prompts_dict and ordered_prompts_lbs_dict.
#         '''
        
        
#         deleter = lambda l, id_to_del : [i for j, i in enumerate(l) if j not in id_to_del]

#         for ptype, plist in ordered_prompts_dict.items():
            
#             #We need to check that the ptype is even being used first and foremost.
#             if ptype in self.valid_ptypes: 

#                 #Any removal must occur on both the prompts and the prompt labels, so we save indices of repeats. We perform this check up to the index to reduce 
#                 # check operations (and since we will not worry ourselves about overwriting points [unlike mistakes simulation].)
#                 removal_indices = {i for i, v1 in enumerate(plist) if any(torch.all(v1 == v2) for v2 in plist[:i])}

#                 new_plist = deleter(plist, removal_indices)
#                 new_plb_list = deleter(ordered_prompts_lb_dict[f'{ptype}_labels'], removal_indices)

#                 ordered_prompts_dict[ptype] = new_plist
#                 ordered_prompts_lb_dict[f'{ptype}_labels'] = new_plb_list 

#         return ordered_prompts_dict, ordered_prompts_lb_dict 

#     def init_sample_regions_components(
#                             self, 
#                             pred: Union[None, Union[torch.Tensor]], 
#                             gt: Union[torch.Tensor],
#                             ):
#         '''
#         Function which initialises the prompt sampling regions on a component basis. Returns a dict of regions 
#         for gt split by class and components, and also the same for the error regions, but split according to the
#         gt and preds.

#         For instances where the error region is not desired (initialisation), the error region subdict is None.
    

#         Motivation: Class specific challenges wrt ordering the components. E.g. if we had a huge organ and a small
#         organ both oversegmented into background, the strategy of only using false negatives would bias to the huge 
#         organ. Similarly, there may be variation in component sizes within a class.

#         Resolution- 
        
#         For the error region: 
        
#         Split into class, then split each false negative region by the predicted class (this would be a
#         false positive of that class). Then into a list of components (torch tensors) within that predicted class.
        
#         Then, apply a sorting algorithm during the iterative looping for each heuristic.

#         For the gt: Split into class, then into a list of components (torch tensors) within the class. 



#         inputs: 
        
#         pred - Optional (torch.Tensor or MetaTensor) discretised pred (not one-hot encoded)
#         gt - torch.Tensor or MetaTensor which is discrete and not one-hot encoded.

#         returns:
        
#         sampling_regions_dict: A dictionary, contains two fields:

#         'gt': Contains a dictionary which is structured as: dict[class_str, list of torch.Tensors]
        
#         'error_regions': Contains a nested dictionary which is structured as:

#         dict[class_str denoting the GT of error voxel, dict[class_str denoting the prediction, list of torch.Tensors]]
        
#         This dictionary chunks up the false negative error region in the following manner: It splits it by the 
#         ground truth class of the error voxels, it then splits it according to the predicted class, it then splits it
#         into a list of components from the error voxels extracted from this intersection.  

#         NOTE: For any instances where a gt class is empty, or the error region is empty for the given config path, then 
#         it will be an empty list []. 
#         '''
#         raise NotImplementedError('Needs to be checked for debugging again, also add a logical check e.g. that gts sum to quantity of voxels')
    
    
#     def init_sample_regions_no_components(
#                             self, 
#                             pred: Union[None, Union[torch.Tensor, MetaTensor]], 
#                             gt: Union[torch.Tensor, MetaTensor],
#                             ):
#         '''
#         Very basic function which initialises the prompt sampling regions without a per-component basis. 
#         Returns a dict of  regions for gt split by class, and also the same for the error regions.

#         It is not intended for sophisticated behaviours, e.g. where one must consider the per-component basis, or the
#         scale of a class. 

#         For instances where the error region is not desired (initialisation), the error region subdict is None.
#         Otherwise, it just treats the entirety of the error region as a single mask for each class it belongs to.

#         inputs: 
        
#         pred - Optional (torch.Tensor or MetaTensor) discretised pred (not one-hot encoded)
#         gt - torch.Tensor or MetaTensor which is discrete and not one-hot encoded.

#         returns:
        
#         sampling_regions_dict: A dictionary, contains two fields:

#         'gt': Contains a dictionary which is structured as: dict[class_str, torch.Tensor]
        
#         'error_regions': Contains a nested dictionary which is structured as:

#         dict[class_str denoting the GT of error voxel, torch.Tensor]]
        
#         This dictionary chunks up the false negative error region in the following manner: It splits it by the 
#         ground truth class of the error voxels, it then splits it according to the predicted class. 

#         NOTE: For any instances where a gt class is empty, or the error region is empty for the given config path, then 
#         it will be Nonetype!  
#         '''
#         sampling_regions_dict = dict() 
        
#         #First we implement the gt extraction since this is always required. 

#         if not isinstance(gt, torch.Tensor) and not isinstance(gt, MetaTensor):
#             raise Exception('The ground truth must be a torch tensor or a metatensor.')
#         #Place gt on device and in int8 dtype. 

#         if not gt.device == self.sim_device:
#             warnings.warn('The gt mask must be placed on the sim device')
#             gt = gt.to(dtype=torch.int8, device=self.sim_device)
        
#         sampling_regions_dict['gt'] = dict.fromkeys(self.semantic_id_dict.keys(), None)

#         #We then split the gt, by class for each class. 
#         accum = None
#         for label, value  in self.semantic_id_dict.items():
#             #We split gt by label. 
#             if not (gt == value).sum(): #0 evaluates to bool False.
#                 warnings.warn(f'Class {label} was empty in gt.')
#                 sampling_regions_dict['gt'][label] = None 
#             else:
#                 sampling_regions_dict['gt'][label] = gt == value
            
#                 if accum is None:
#                     accum = sampling_regions_dict['gt'][label]
#                 else:
#                     #We cumulatively add the gt regions, checking for overlaps with the current region under consideration.
#                     if (accum & sampling_regions_dict['gt'][label]).sum():
#                         raise Exception(f'Overlap detected between gt regions') 
#                     #If it passes the overlap check, we add to the accum for the next region check.
#                     accum = accum | sampling_regions_dict['gt'][label]

#         accum = accum.to(device='cpu')
#         torch.cuda.empty_cache()
#         del accum

#         if all([masks is None for label, masks in sampling_regions_dict['gt'].items() if label.title() != 'Background']):
#             raise Exception('All of the foreground classes in the ground truth cannot be empty.')
        
#         #Here we will be checking that the GT-split region has no overlaps and is inclusive of the entire image.

#         # if torch.all(torch.stack([i for i in sampling_regions_dict['gt'].values() if i is not None]).sum(dim=0) == torch.ones_like(gt)):
#         #     print('GT inclusive of background merged to a tensor of ones.')
#         # else:
#         #     raise Exception('GT maps did not merge to a tensor of ones')



#         #if pred is None, then we just return the gts.
#         if pred is None:
#             sampling_regions_dict['error_regions'] = None 
#             return sampling_regions_dict #No need to calculate any error regions as there are no predictions!
#         else: 
#             #Place pred on device and in int8 dtype (not bool yet!). 

#             if not pred.device == self.sim_device:
#                 warnings.warn('The pred mask must be placed on the sim device')
#                 pred = pred.to(dtype=torch.int8, device=self.sim_device)
            
#             #Find the false negative error region. 
#             # error_map_bool = torch.where(pred != gt, 1, 0).to(dtype=torch.int8, device=self.sim_device)
#             error_map_bool = pred != gt #We want it in bool format to minimise memory usage! Same memory usage as int8 though.
            
#             #Create the error regions dict: 
#             err_regions_dict = dict() 

#             accum = None
#             for l1, v1 in self.semantic_id_dict.items():
#                 #Splitting into classes according to gt (i.e. voxels where an error occured and where the gt class exists)                
#                 # temp_gt = torch.where(gt == v1, 1, 0).to(dtype=torch.int8, device=self.sim_device) #DEPRECATED NOT NEEDED USE OF MEMORY!

#                 #We split error region by gt class, and because calling from gt above NoneType would break we also recompute the class-
#                 # separated gt map.
#                 # split_by_gt = error_map_bool & (gt == v1)  
#                 # if split_by_gt.sum() == 0:
#                 # if not split_by_gt.sum():
#                 if not (error_map_bool & (gt == v1)).sum():
#                     err_regions_dict[l1] = None 
#                 else: 
#                     # err_regions_dict[l1] = split_by_gt
#                     err_regions_dict[l1] = error_map_bool & (gt == v1) 
#                     if accum is None:
#                         accum = err_regions_dict[l1]
#                     else:
#                         #We cumulatively add the error regions, checking for overlaps with the current region under consideration.
#                         if (accum & err_regions_dict[l1]).sum():
#                             raise Exception(f'Overlap detected between error regions') 
#                         #If it passes the overlap check, we add to the accum for the next region check.
#                         accum = accum | err_regions_dict[l1]


#             #Dumping VRAM as quickly as possible. 
#             accum = accum.to(device='cpu')
#             error_map_bool = error_map_bool.to(device='cpu')
#             del accum, error_map_bool
#             torch.cuda.empty_cache()

#             sampling_regions_dict['error_regions'] = err_regions_dict 

#             #We no longer need the gt and the pred. Lets dump VRAM as quickly as possible. 
#             pred = pred.to(device='cpu')
#             gt = gt.to(device='cpu')
#             del pred, gt, err_regions_dict
#             torch.cuda.empty_cache()


#             #Checking the error regions:
#             if all([masks is None for masks in sampling_regions_dict['error_regions'].values()]):
#                 raise Exception('All of the error regions cannot be empty, should have terminated the iterative loop already..')

#             #Extracting the valid (non-nonetype) error regions across all classes. (i.e. the general error region), if all error regions were None then it should break already.
#             # err_regions_check = torch.stack([i for i in sampling_regions_dict['error_regions'].values() if i is not None]).sum(dim=0)
#             # if torch.all((err_regions_check == 0) | (err_regions_check == 1)):
#             #     print('No error regions are overlapping as the  values of the sum across classes for the error region maps is 0 or 1 for each voxel!')
#             # else:
#             #     raise Exception('There are overlapping error regions')

#             # err_regions_check = err_regions_check.to(device='cpu')
#             # torch.cuda.empty_cache()

#             # del err_regions_check

#             return sampling_regions_dict

#     def update_error_region(self, region_mask, prompts: list[torch.Tensor], prompt_type: str):
#         '''
#         This is a function which updates a region mask according to a set of prompts.

#         This can be incorporated into an approach for multi-component handling, multi-class handling and also for 
#         handling different prompt types.

#         In particular, it can handle points and scribbles under the umbrella of free-form prompts. And box/lasso regions
#         under the umbrella of region-based prompts. It assumes that the prompts are provided as a list of
#         tensors with shape N x N_dim (N = 1 for points, and N_s.p for scribbles). It will raise an exception if
#         the spatial dimensions are larger than N_dims. 

#         inputs:

#         region_mask: A binary mask with N_dim spatial dims denoting an error region with values of 1, everywhere else is zero. 
#         prompts: A list of prompts N x N_dim for updating the region mask.
#         prompt_type: The name of the prompt type: points, scribbles, bboxes, lassos.
#         '''
#         # if not all([prompt.shape[1] == region_mask.dim() for prompt in free_form_prompts]):
#         #     raise Exception('The number of spatial dimensions of all input prompts must match the number of spatial dimensions of the mask')
    
#         if prompts == []:
#             warnings.warn('Trying to update the sampling region but the inserted prompts are empty. Check that this is valid.')
#             return region_mask

#         if prompt_type in self.free_form_prompts_ls:
#             #Instead, we will fuse all of the coordinates, hence permitting for the handling to occur in a single step after merging.
#             coords = torch.cat(prompts, dim=0)
#             if not coords.shape[1] == region_mask.dim():
#                 raise Exception('The spatial dimensions of the input prompts must match the number of spatial dimensions in the mask.')

#             if region_mask.device != self.sim_device:
#                 region_mask.to(device=self.sim_device)

#             # for coords in free_form_prompts:  
#             region_mask = update_binary_mask_freeform(coords, region_mask)

#         elif prompt_type in self.partition_prompts_ls:
#             raise NotImplementedError('Partition based prompt region updating not implemented yet.') 
#         else:
#             raise Exception(f'Prompt subtype {prompt_type} not recognised for region updating.')
#         return region_mask
    
#     def sort_components(self, components: Union[list[Union[torch.Tensor]], None], sort_criterion: str = None):
#         '''
#         This is a function that will sort a list of mask components according to a sort criterion. 

#         inputs: 

#         components - list: A list of components, can optionally also be an empty list.
#         sort_criterion - str: A sorting criterion.
#         '''

#         if sort_criterion is None:
#             return components 
#         elif sort_criterion.title() == 'Random':
#             return self.shuffle_list(components, 'random')
#         elif sort_criterion.title() == 'Component Sum':
#             #This is on the basis of the sum of the torch tensors (i.e. the size of the component)
#             _, indices = torch.sort(torch.stack(components).sum(list(range(1, len(torch.stack(components).shape)))))
#             return [components[i] for i in indices]


# class SimplifiedPrototypePseudoMixture(BasicValidOnlyMixture):
#     '''
#     #NOTE: Much of the implementation here is not required for the prototype, but is left in place for future reference where
#     # more complex mixture models may be implemented. For example, the use of region in-filling strategies is not required
#     # if we restrict per-iteration simulation to 1 prompting strategy/heuristic for a single prompt type (from the selection of methods configured), or if we do not
#     # have complex inter/intra prompt level interactions). Whereas it would be required for complex mixture models or cases
#     # when a prompt type may have multiple heuristics used for simulation in a single iteration.

#     This class implements a prototype for iterating through the prompt-gen heuristics fns to 
#     simulate prompts.

#     Intended for heuristics implementations which do not have complex mixture args.
      
#     Plain class-level handling. 
#     Inter-prompt level handling is restricted only to permitting one prompt type to be sampled per iteration.  
#     Intra-prompt level handling is only restricted to sampling without replacement. 
#     Heuristic handling is only restricted to basic configurable arguments for a heuristic, not whether the heuristic is used 
#     or not/drop-out. 


#     (E.g. No toggling of the drop-out of prompts, no toggling of the order in which prompts are generated (just does it randomly), heuristics order, etc., 
#     components order doesn't even exist it is treated as a singular error map which is handled by heuristics directly. 
    
#     Toggling off the use-mem for determining whether im is used for prompt generation. 
    
#     The only cross-interaction is the removal of coordinates for sampling prompts on an inter-prompt basis.

#     '''
#     def __init__(
#             self,
#             semantic_id_dict: dict,
#             sim_device: torch.device,
#             heur_fn_dict: dict,
#             build_args: dict,
#             mixture_args: dict = None,
#             use_mem: bool = False,
#             ):
        
#         super().__init__(
#             use_mem=use_mem,
#             semantic_id_dict=semantic_id_dict,
#             sim_device=sim_device,
#         )
#         self.heur_fn_dict = heur_fn_dict

#         #Inter-prompt level default variable.
#         self.prompt_level_order = [['bboxes', 'lassos'], ['scribbles', 'points']]

#         #List denoting the priority list of prompt types.it bins the prompt types into distinct groups of priority, 
#         # each sublist has items more equal in priority. This is created in order to facilitate the prompt sampling process,
#         # where partition based prompts are capable of splitting image, whereas free-form prompts cannot. Generally we want to
#         # sample without replacement (i.e., it is fairly reasonable to expect a user to not overwrite the same coordinates within
#         # a single iteration). Hence, we prioritise partition based prompts first, as these will enclose a region. 
        
#         #Denoting variables for partition, and free-form prompts. partition prompts partition the image space into inside-outside
#         # regions, while free-form prompts provide no prescriptive description of inside-outside. Only: look here!

#         #Variables for denoting the partition and free-form type prompts
#         self.partition_prompts_ls = ['bboxes', 'lassos']
#         self.free_form_prompts_ls = ['points', 'scribbles']

#         #Initialising the list of valid prompt types.
#         self.init_valid_ptypes(build_args=build_args)

#         #Initialising the toggling dict which provides the information necessary for toggling throughout the cascade.
#         self.init_toggle_dict(heur_build_args=build_args, mixture_args=mixture_args)
        
#         #Checking that the heuristic level params are actually supported.
#         self.check_heur_params() 

#     def check_heur_params(self):
        
#         for ptype, heurs_configs in self.toggling_dict['intra_heur_level'].items(): 

#             if ptype in self.valid_ptypes:
#                 if heurs_configs is None:
#                     raise Exception('The heuristic params cannot be a NoneType if we are simulating for a given prompt')
#             else:
#                 if heurs_configs is not None:
#                     raise Exception('Attempted to provide heuristics configuration for a non-valid prompt type.')
              
                 
#     def init_valid_ptypes(self, build_args: dict, simulation_type: str = 'simplified_prototype'):
#         '''
#         Function which extracts the list of valid prompt types according to the build args dict.
#         '''
#         #Populate the list of valid (used/configured) prompt types according to the dict. 
        
#         #First checking that all of the prompt types have been configured in some capacity (even if NoneType) according to
#         # a reference of configurations. In this case, just the heuristic functions dictionary.
#         self.check_config_availability({'heur_params': build_args}, prompter_type=simulation_type)

#         #Checks whether the heur function dict is a Nonetype by default.
#         self.valid_ptypes = [key for key,val in build_args.items() if val is not None]

#         if len(self.valid_ptypes) != 1:
#             raise Exception('Exactly one valid prompt type must have been configured! We do not support any cross-interactions '
#             'between prompt types in the prototype pseudo-mixture model.')
        
#         if 'scribbles' in self.valid_ptypes or 'bboxes' in self.valid_ptypes or 'lassos' in self.valid_ptypes:
#             raise NotImplementedError('We have selected bbox, scribbles, or lassos in the prompt gen. configs but they are ' \
#             'not implemented in the utilities currently.')

#     def init_prompts(self):
#         '''
#         Function which initialises the prompts and prompt labels dictionary according to the valid prompt types 
#         (and also cross-references this again against the heuristics function dict).

#         Returns:

#         tracked_prompts: A dictionary, split by prompt type, which contains the initialised dict according to the 
#         set of valid prompt types (i.e. those for which a prompt can be simulated). Contains empty lists for valid,
#         and NoneTypes for invalid prompt types.

#         tracked_prompts_lbs: Same as tracked_prompts, except for the prompts' labels. Split by {prompt_type}_labels
#         '''
#         #We initialise the dictionary containing the prompts and labels across the prompt types with val None:
#         prompts = dict.fromkeys(self.heur_fn_dict.keys(), None)
#         prompt_lbs = dict.fromkeys([i + '_labels' for i in self.heur_fn_dict.keys()], None)
    
#         for ptype in self.valid_ptypes:
#             #Populate the list of valid (used/configured) prompt types according to the heuristics dict. 
            
#             #Checks again whether the heuristics dict is a Nonetype by default.
#             if self.heur_fn_dict[ptype] is not None:

#                 #Initialises with a list for the valid ptypes 
#                 prompts[ptype] = []
#                 prompt_lbs[f'{ptype}_labels'] = []

#         #Check that the initialisations are indeed NoneTypes for the non-valid ptypes, and empty lists otherwise.
#         if not all([vp is None and prompt_lbs[f'{k}_labels'] is None for k,vp in prompts.items() if k not in self.valid_ptypes]):
#             raise Exception('The initialised prompts and prompt labels were invalid for the non-valid prompt-types.')
#         if not all([vp == [] and prompt_lbs[f'{k}_labels'] == [] for k,vp in prompts.items() if k in self.valid_ptypes]):
#             raise Exception('The initialised prompts and prompt labels were invalid for the valid prompt-types.')

#         return prompts, prompt_lbs

#     def init_toggle_dict(self, 
#                         heur_build_args: dict,
#                         mixture_args: Union[dict, None]):
        
#         if heur_build_args is None:
#             raise Exception('Heuristic level build arguments are ALWAYS required. At least one per heuristic!')
#         elif heur_build_args is not None and mixture_args is None:
#             #In this case, we will resort to defaults for the mixture methods.
#             self.toggling_dict = {
#                 'class_level': None, 
#                 #None = Just use a default provided.
#                 'inter_prompt_level': None, 
#                 #None = Just use a default provided. 
#                 'intra_prompt_level': None,
#                 #None = Just use a default provided.
#                 'intra_heur_level': heur_build_args,
#                 #Use the heuristic build args provided.
#             }
#         else:
#             raise NotImplementedError('Not implemented anything for handling the toggling of anything non-default wrt mixture strategies.')
        
#     def togg_class_level(self, 
#                         tracked_prompts, 
#                         tracked_prompt_lbs, 
#                         samp_regions_dict,
#                         init_bool):
#         '''
#         Executes the prompt simulation process by the class level toggling using the toggling dictionary.

#         Inputs:
        
#         tracked_prompts: P-type separated dict (The initialised tracking prompts which will be tracked throughout)
        
#         tracked_propmt_lbs: P-type separated dict (The initialised tracking prompt labels which will be tracked throughout)
        
#         samp_regions_dict: The nested dictionary (split by gt and error region) denoting the class separated regions (or Nones for empty gt/error regions for a given class)
        
#         init_bool: A bool, denotes whether the inference call the prompt generation is for is an init or edit, this
#         is required for downstream in order to delineate between instances where prompts are being placed on gt or 
#         error region. Required because we cannot infer reliably from the datatype of the error-region and gt after
#         we pass deeper past the class-level toggling.

#         (e.g., Error-region separated on class level could have a NoneType for a class because it is empty, or it 
#         could just be because the error-region entry empty because it is an initialisation, the GT must always be not a 
#         nonetype at that level, otherwise there would be no error-region anyways!)

#         '''
        
#         if self.toggling_dict['class_level'] is None:
#             #None = default behaviour.
#             if samp_regions_dict['gt'] is None:
#                 raise Exception('The entire ground truth cannot be a NoneType..otherwise we cannot even sample.')
#             #Checking that at least one foreground class has a GT..., should already be handled in the front-end but just in case!
#             if all([val is None for key,val in samp_regions_dict['gt'].items() if key.title() != 'Background']):
#                 raise Exception('Error in code, no foreground gt available, should have been flagged earlier?')
             
#             #Checks in place for handling init/edit.
#             if not init_bool:
#                 if samp_regions_dict['error_regions'] is None: 
#                     raise Exception('Cannot have a nonetype for error region dictionary if simulating edit') 
                    
#                 if all([item is None for item in samp_regions_dict['error_regions'].values()]):
#                         #If all the error regions for all classes are NoneTypes
#                         raise Exception('Error in code, no errors remain, should have exited the iterative loop simulation on full convergence, and should have flagged this in the error region extraction phase.')
#             else:
#                 if samp_regions_dict['error_regions'] is not None:
#                     raise Exception('Cannot have a non-Nonetype for error region item if simulating initialisation.') 
                
             
#             for class_lb, class_int in self.semantic_id_dict.items():
                
#                 #By default, We just iterate through on a class by class basis. 

#                 #We, do not elect to return an updated sampling region (updated by the prompts placed),
#                 # as this functionally does nothing for the current prototype (we are performing on a 
#                 # class-by-class basis without errors simulated). 

#                 # print(f'Sampling prompts in class {class_lb} \n')
#                 if samp_regions_dict['gt'][class_lb] is None: 
#                     #We implement a check here to see if we can skip over..
#                     # print(f'Skipping class {class_lb} as it has no gt and (or by extension) false-negative error region \n')
#                     continue 
#                 else:
#                     #Here we extract the gt and the error region for the given class.
#                     if init_bool:
#                         #If initialisation, then we only have access to the ground truth region for the current class.
#                         regions_dict = {
#                         'gt':samp_regions_dict['gt'][class_lb],
#                         'error_regions': None
#                         }
#                     else:
#                         #If edit, extract the gt and the false negative error region for the current class.
#                         if samp_regions_dict['error_regions'][class_lb] is None:
#                             print(f'Skipping class {class_lb} for editing as it has no false negative error region, hence no free-form prompts can be placed (necessary) \n')
#                             if set(self.valid_ptypes) & set(self.partition_prompts_ls) != set():
#                                 raise Exception('We still have not fixed the handling of partition prompts, and so we cannot skip over if we use partition prompts! Current approach requires sampling at every iteration')
#                             else:
#                                 continue 
#                         else:
#                             regions_dict = {
#                                 'gt':samp_regions_dict['gt'][class_lb], 
#                                 'error_regions':samp_regions_dict['error_regions'][class_lb] 
#                                 #NOTE: There is currently a potential logical conflict. 
#                                 # We should theoretically not even require the gt for editing, but we had included it temporarily until we 
#                                 # decided how to handle partition (i.e. non-editing) prompts. 
#                                 } 

#                 gen_prompts = self.togg_inter_prompt_level(
#                     samp_regions_dict=regions_dict,
#                     init_bool=init_bool
#                 )

#                 for ptype in self.valid_ptypes:
                    
#                     assert type(gen_prompts[ptype]) == list, 'Generated prompts, even if empty, must be a list'

#                     #We skip over the non-valid ptypes.
#                     temp_plist = copy.deepcopy(tracked_prompts[ptype])
#                     temp_plist.extend(gen_prompts[ptype])
                    
#                     temp_plab_list = copy.deepcopy(tracked_prompt_lbs[f'{ptype}_labels'])
#                     #Just create a list according to the generated prompts. If it is empty (i.e. len of 0 then will just extend by [])
#                     gen_prompts_lbs = [torch.tensor([class_int], dtype=torch.int8, device=self.sim_device)] * len(gen_prompts[ptype])

#                     temp_plab_list.extend(gen_prompts_lbs)

#                     tracked_prompts[ptype] = temp_plist
#                     tracked_prompt_lbs[f'{ptype}_labels'] = temp_plab_list


#             return tracked_prompts, tracked_prompt_lbs

#         else:
#             raise Exception('No other class-level toggling methods have been implemented yet other than the default.')    
#     def togg_inter_prompt_level(self, 
#                                 samp_regions_dict,
#                                 init_bool):
#         '''
#         This executes handling at the inter-prompt level. Any cross-interactions at an inter-prompt level should be
#         handled here. By default, we assume no inter-prompt level interactions. 

#         Inputs: 

#         samp_regions_dict: Dictionary of sampling regions for the current class in the parent toggle 
#         (toggle_class_level). Contains the 'gt' and the 'error_region' (gt can never be a NoneType, error region CAN).
#         init_bool: A bool denoting whether the prompt generation is for an initialisation or not (relevant for 
#         downstream toggles handling)

#         returns: Generated prompts (spatial coords) (denoted within this function as tracked prompts as it will be
#         tracking across the loops). A dictionary, separated by prompt type, containing either lists of tensors or 
#         Nones for the merging at the class-level. 
#         '''
#         if samp_regions_dict['gt'] is None:
#             #If the gt is empty then break, this should have been flagged at the class level.
#             raise Exception('Somehow an empty gt class got through, check the code logic')

#         if self.toggling_dict['inter_prompt_level'] is None:
#             #None = default behaviour. 

#             #We initialise the tracked prompts. We set NoneTypes for non-valid explicitly to help flag any errors. 
#             tracked_prompts = dict.fromkeys(self.free_form_prompts_ls + self.partition_prompts_ls, None)
#             tracked_prompts.update(dict.fromkeys(self.valid_ptypes, [])) 
            
#             #We define sampling regions depending on the prompt-type category and whether it is an init or an 
#             # editing prompt. 

#             # NOTE: DEPRECATED: For partition prompts, this was previously exclusively simulated using the ground truth. 
#             # This is no longer the case, as we may want to simulate partition prompts on error regions, even for bbox! 
#             # partition_region = samp_regions_dict['gt'] 
            
#             #Depending on whether it is an initialisation or an editing prompt, the reference region will change.
#             if init_bool:
#                 #We use deepcopies to prevent potential leakage #that could occur due to variable assignments.
#                 region = copy.deepcopy(samp_regions_dict['gt'])
#                 # freeform_region = samp_regions_dict['gt']
#             else:
#                 #Editing prompts, use the error regions.
#                 if samp_regions_dict['error_regions'] is None:
#                     #In this case, there is no error for this class! We cannot place anything. 
#                     #NOTE: We do raise an exception because for an editing iteration we need something, should have exited out already.
#                     raise Exception('Error, we cannot place prompts for an error region which is empty, this should have been handled at the class-level')    
#                 else:
#                     #Otherwise, use the error region!
#                     # partition_region = samp_regions_dict['error_regions']
#                     # freeform_region = samp_regions_dict['error_regions'] 
                
#                     #We use deepcopies to prevent potential leakage #that could occur due to variable assignments.
#                     region = copy.deepcopy(samp_regions_dict['error_regions']) 

        
#             #Simplified version of this function for the prototype, as we enforce only one prompt type to be used. This 
#             #will have a knock-on effect with random seeds. Hence why we mostly duplicated the code. 

#             ##################################################################################################
#             ptype = self.valid_ptypes[0] #There is only one valid ptype in the prototype.
        
#             print(f'Simulating prompts for prompt type: {ptype} \n ')
                
#             if region.dtype != torch.bool:
#                 raise TypeError('Sampling region masks must be of type torch.bool') 
#             #We have already checked that the gt region is not empty at the start of this function.
            
#             if region is None:
#                 #In this case, there was nowhere to place prompts for this class!
#                 raise Exception('Hello? this should never happen. Pay attention designated programmer.')
#             else:
#                 #We do not need to update the region as there is only one prompt type per iteration in the prototype.

#                 #Here we pass through the sampling region
#                 ptype_gen_prompts = self.togg_intra_prompt_level(
#                 ptype=ptype,
#                 samp_region=region,
#                 # init_bool=init_bool
#                 )

#                 if not isinstance(ptype_gen_prompts, list):
#                     raise TypeError('The output of the intra-prompt level function must be a list of prompts.')
                
#                 #Here we merge generated prompts with the tracked prompts. We are already in an if-else 
#                 #condition wrt valid ptypes. So no check is required.
                
#                 # NOTE: any empty lists will still be completely valid as they can be handled
#                 #at the class-level. Moreover, a check is implemented in the heuristics builder to ensure 
#                 #at least one valid prompt (and free-form prompt) is generated.

#                 tracked_prompts[ptype] = ptype_gen_prompts
                    
#         else:
#             raise Exception('Inter-prompt level toggling other than the default is not implemented for the prototype, the ' \
#             'current simulation strategy is a simple heuristic without cross-interactions and restricted to single '
#             'prompt types.')
                       
#         return tracked_prompts
    
#     def togg_intra_prompt_level(self, 
#                                 ptype:str,  
#                                 samp_region:Union[torch.Tensor]
#                                 ):
#         '''
#         Function which iterates through the heuristics for each prompt type at the intra-prompt level
         
#         Inputs: 
    
#             ptype - str: The prompt type 
#             samp_region - Torch Tensor: The tensor containing a binary mask for the sampling region.

#         Returns:
#           generated_prompts (denoted as tracked_prompts in the function) 
          
#           A list of generated prompts for the given ptype using the corresponding heuristics provided.
#         '''

#         #Only gets triggered for valid ptypes in the parent toggle level (inter-prompt toggle). 

#         #Checking whether we have anything to even sample from:
#         if samp_region.dtype != torch.bool: #We require bool types, as our downstream checks are dependent on this.
#             raise TypeError('Sampling region tensor must be of type torch.bool')
#         if samp_region is None or not samp_region.sum(): #samp_region != 0 will evaluate to a bool=True. 
#             #torch.all(samp_region == torch.zeros_like(samp_region)):
#             #Sampling region should have been flagged.
#             raise Exception('Somehow an empty sampling region got through to the intra-prompt level, please check the code logic')


#         if self.toggling_dict['intra_prompt_level'] is None:
#             #None = Default behaviour for the prototype. 

#             #We initialise a list for the prompts:
#             tracked_prompts = []

#             #We extract the heuristics dictionary for the given prompt type.
#             heurs_dict = self.heur_fn_dict[ptype]

#             #We shuffle the heuristics list randomly for ensuring prompting diversity.

#             # E.g., any sampling without replacement will inherently be conditioning the prompt generation by 
#             # affecting the sampling region.
            
#             shuffled_heurs_order = self.shuffle_list(list(heurs_dict.keys()), 'random')
            
#             for heur in shuffled_heurs_order:
#                 #Honestly... what the hell was I thinking when I wrote the initial code for this. Why would I update with an empty list to begin with.
#                 print(f'Filling in the sampling region for sampling without replacement: {ptype} at the intra-prompt level \n')
#                 #We are sampling without replacement at the intra-prompt level. 
#                 if ptype in self.free_form_prompts_ls or ptype in self.partition_prompts_ls:
#                     region = copy.deepcopy(samp_region)
#                     #NOTE: We do this from scratch each time just to be sure that there is no leakage as the 
#                     # update error will modify the mask in place permanently. It may not be the case that we are sampling with replacement
#                     # across prompt types, but it is better to keep these isolated. 

#                     ##TODO: Managing these sampling-regions will be key for further VRAM optimisation!
#                     #NOTE: The update error region function can handle empty lists!
#                     region = self.update_error_region(
#                         region_mask=region, 
#                         prompts=tracked_prompts, 
#                         prompt_type=ptype)
#                 else:
#                     raise Exception('Prompt type does not fall under the partition or free-form prompt types. How did we get here?')
                
#                 #If the sampling region in-filled/filtered is zeroes then we must terminate, no more prompts can 
#                 # be placed. We do not check nonetypes because nonetypes should never be sampled nor passed through! 
#                 if region.dtype != torch.bool:
#                     raise TypeError('Sampling region masks must be of type torch.bool')
                
#                 # if torch.all(region == torch.zeros_like(region)):
#                 if not region.sum():#If the sum is zero, then there is no region to sample from, so early termination for prompt gen.
#                     print(f'Early termination of the prompt generation for ptype: {ptype} \n')
#                     break 
#                 else:
#                     generated_prompts = self.togg_intra_heur_level(
#                         ptype=ptype,
#                         heur=heur,
#                         samp_region=region 
#                     )
#                     tracked_prompts.extend(generated_prompts)  
#         else:
#             raise Exception('Intra-prompt level toggling other than the default is not implemented for the prototype. The default \n' \
#             'is sampling without replacement across heuristics for a given prompt type across all configured heuristics. There is no' \
#             'dropout or complex interactions implemented.')
#         return tracked_prompts 
    
#     def togg_intra_heur_level(self, 
#                         ptype: str,
#                         heur: str, 
#                         samp_region: Union[torch.Tensor]):
#         if samp_region.dtype != torch.bool:
#             raise TypeError('Sampling region tensor must be of type torch.bool')
        
#         # if samp_region is None or torch.all(samp_region == torch.zeros_like(samp_region)):
#         if samp_region is None or not samp_region.sum(): #If the sum is zero, then there is no region to sample from.
#             raise Exception('The sampling region needs to be able to be sampled, it is empty or None!!')
        
#         if self.toggling_dict['intra_heur_level'] is None:
#             raise Exception('There must be at the very minimum some heuristic level toggling/args otherwise we cannot call on abstract heuristics.')
#             #Default behaviour requires heuristic level arguments so that there is a function to call on for generating a prompt.
#         else:
#             #Else, then just extract the heuristic, and the params.
#             heur_fnc =  self.heur_fn_dict[ptype][heur]
#             params = self.toggling_dict['intra_heur_level'][ptype][heur]

#             if ptype == 'bboxes' or ptype == 'scribbles' or ptype == 'lassos':
#                 raise NotImplementedError('We should not have reached ptypes of bbox, scribbles, or lassos yet, they are not supported!')
          
#             generated_prompt = heur_fnc(samp_region, params)
#             if not isinstance(generated_prompt, list):
#                 raise Exception('The generated prompt must always be a list, even if it is empty!')
            
#             return generated_prompt
        
#     def __call__(self, data):
#         '''
#         Function which calls on the methods for implementing the prompt generation process. 

#         inputs: 

#         data: A dictionary containing the following fields: 

#         image: Torch tensor OR Metatensor containing the image in the pseudo-ui image domain (no pre-processing applied other than re-orientation in RAS)
#         gt: Torch tensor OR Metatensor containing the ground truth map in RAS orientation, but otherwise in the pseudo-ui image domain (no pre-processing other than RAS re-orientation).

#         prev_output_data: (NOTE: OPTIONAL, is NONE otherwise) output dictionary from the inference call which has been post-processed 
#         in the pseudo-ui front-end.
       
#         Two relevant fields for prompt generation contained are the: 
#             pred: A dictionary containig two relevant subfields
#                 1) "metatensor" A Metatensor or torch tensor (1HW(D)) containing the previous segmentation in the pseudo-ui image domain (no pre-processing applied other than re-orientation in RAS) 
#                 2) "meta_dict" A dict containing (at least) the affine matrix for the image, containing pseudo-ui image domain relevant knowledge.

#             probs: A dictionary containing two relevant subfields
#                 1) "metatensor" A Metatensor or torch tensor (CHW(D)) containing the previous segmentation in the pseudo-ui image domain (no pre-processing applied other than re-orientation in RAS) 
#                 2) "meta_dict" A dict containing (at least) the affine matrix for the image, containing pseudo-ui image domain relevant knowledge.
        
#         im: Optional (or NoneType) dictionary containing the interaction memory from the prior interaction states.      
#         '''

#         if self.use_mem:
#             #Extract the interaction memory.
#             im = data['im']

#             if not im:
#                 raise Exception('If using interaction memory, then it requires interaction memory available! Received nonetype')
            
#             raise NotImplementedError('Not permitting the use of interaction memory in the prototype prompt generator, no memory conditioning.') 
#         else:
#             if data['prev_output_data'] is None:
#                 print('We have no prior output data, please check that this is an initialisation! \n')
#                 # pred = None
#                 init_bool = True 

#                 if data['im'] is not None:
#                     raise Exception('The interaction memory should be a NoneType for the initialisation.')
#             else:
#                 print('We have prior output data, please check that this is an editing iteration \n')
#                 # pred = data['prev_output_data']['pred']['metatensor'][0, :]
#                 # pred = pred.to(dtype=torch.int8, device=self.sim_device)

#                 # gt = data['gt'][0,:].to(dtype=torch.int8, device=self.sim_device)
#                 if not (isinstance(data['prev_output_data']['pred']['metatensor'][0, :], torch.Tensor) or isinstance(data['prev_output_data']['pred']['metatensor'][0, :], MetaTensor)):
#                     raise TypeError('The pred needs to be a torch tensor or a Monai MetaTensor')            
#                 init_bool = False

#                 if data['im'] is None:
#                     raise Exception('The interaction memory (even if unused) should not be a NoneType for edits.')
                
#             if not isinstance(data['gt'], MetaTensor):
#                 raise TypeError('The gt needs to be a Monai MetaTensor')
            
#             #Extracts a dict with fields 'gt' and 'error_regions'. Both class separated dicts.
#             sampling_regions_dict = self.init_sample_regions_no_components(
#                 pred=data['prev_output_data']['pred']['metatensor'][0, :].to(dtype=torch.uint8, device=self.sim_device) if not init_bool else None,
#                 #Loading the gt.
#                 gt = data['gt'][0, :].to(dtype=torch.uint8, device=self.sim_device)
#         )
#             #To prevent VRAM segfault for huge images just in case anything is lingering.
#             torch.cuda.empty_cache() 

#             #We initialise the prompt dictionaries on each call.
#             tracked_prompts, tracked_prompts_lbs = self.init_prompts()

#             #Passing through the initialised prompts through the cascade starts at the class level..
#             tracked_prompts, tracked_prompts_lbs = self.togg_class_level(
#                 tracked_prompts=tracked_prompts, 
#                 tracked_prompt_lbs=tracked_prompts_lbs, 
#                 samp_regions_dict=sampling_regions_dict,
#                 init_bool=init_bool)

#             #Just for assurance we run it through a function which removes any repeats on a intra-prompt level.
#             tracked_prompts, tracked_prompts_lbs = self.rm_intra_prompt_spat_repeats(tracked_prompts, tracked_prompts_lbs)

#             tracked_prompts, tracked_prompts_lbs = self.output_processor(tracked_prompts, tracked_prompts_lbs)
#             return tracked_prompts, tracked_prompts_lbs
 

# prompt_mixture_registry = {
#     'simplified_prototype_pseudomixture': make_factory(SimplifiedPrototypePseudoMixture)
# }