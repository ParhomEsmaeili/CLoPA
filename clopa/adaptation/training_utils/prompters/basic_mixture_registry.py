#This is a file containing prompter registries, in a similar fashion to the more complex formulations (or those which are capable
#of being more complex) such as heuristic mixtures. But completely stripped down for the sake of completing an MVP quickly. 

#Assumed only to be used for binary semantic segmentation tasks, it is much more basic than the more involved mixture registry.

from clopa.adaptation.training_utils.general_utils import make_factory
from clopa.adaptation.training_utils.prompters.utils import update_binary_mask_freeform
import torch
from monai.data import MetaTensor 
from typing import Union
import copy

class RandomAgent:
    def __init__(
            self, 
            args: dict):
        self.sim_device = args.get("sim_device")
        self.semantic_id_dict = args.get("semantic_id_dict")
        self.use_mem = args.get("use_mem")
        heur_fn_dict = args.get("heur_fn_dict")
        build_args = args.get("build_args") #Args for configuring the heuristics
        mixtures_args = args.get("mixture_args") #Args for configuring the mixtures

        if len(self.semantic_id_dict) != 2:
            raise ValueError("The basic random agent prompter only supports binary segmentation tasks. \n"
                             "Please ensure the semantic_id_dict provided contains exactly two entries.")
        if self.sim_device.type != 'cuda':
            raise ValueError("The basic random agent prompter currently only supports CUDA devices. \n"
                             "Please provide a CUDA device.")
        if self.use_mem == None or self.use_mem != False:
            raise ValueError("The basic random agent prompter currently only supports non-memory retention mode. \n"
                             "Please set use_mem to False.")
        if heur_fn_dict == None or not isinstance(heur_fn_dict, dict):
            raise ValueError("The basic random agent prompter requires a heuristic function to be provided. \n"
                             "Please provide a heuristic function.")
        if build_args == None or not isinstance(build_args, dict):
            raise ValueError("The basic random agent prompter requires heuristic parameters to be provided. \n"
                             "Please provide heuristic parameters.")
        if mixtures_args != None:
            raise ValueError("The basic random agent prompter does not support heuristic mixtures. \n"
                             "Please set heuristic_mixtures to None.")
        
        
        self.supported_prompts = ['points', 'scribbles', 'bboxes', 'lassos']
        self.free_form_prompts_ls = ['points', 'scribbles']
        self.partition_prompts_ls = ['bboxes', 'lassos']

        self.discrete_variables = [i + '_labels' for i in self.supported_prompts] #The labels are discrete variables, 
        #the spatial coordinates need not necessarily be discrete. To permit sub-voxel coordinates in future implementations.
        
        #Initialising the list of valid prompt types.
        self.init_valid_ptypes(build_args=build_args)

        self.init_toggle_dict(
            heur_fn_dict=heur_fn_dict,
            heur_build_args=build_args,
            mixture_args=mixtures_args
        )
        
        #Now checkingt that we have configured out toggle dict properly.
        self.check_toggle_dict()

    def check_config_availability(self, input_configs: dict[dict], prompter_type: str = None):
        '''
        Function which loops through a set of configurations which are being utilised in order to check that
        configuration parameters required for configuring the prompt simulation classes are
        provided. I.e., no missing parameters/loose ends. 

        Raises errors if there are supported prompts which have no configuration provided. 

        Currently no support is provided for prompter-type specific configurations, but this is left in place. 
        It just generically checks that all of the supported prompt types have been configured.
        '''
        for config_name, config in input_configs.items():
            for p_type in self.supported_prompts:
                if p_type not in config.keys():
                    raise Exception(f'The prompt type {p_type} was not provided in the configuration dictionary: {config_name} for {prompter_type}') 

    def init_valid_ptypes(self, build_args: dict, simulation_type: str = 'simplified_prototype'):
        '''
        Function which extracts the list of valid prompt types according to the build args dict.
        '''
        #Populate the list of valid (used/configured) prompt types according to the dict. 
        
        #First checking that all of the prompt types have been configured in some capacity (even if NoneType) according to
        # a reference of configurations. 

        #First we check that all of the prompt types are present in the build args dict so we can identify valid ptypes.
        self.check_config_availability({'heur_params': build_args})

        #Checks whether the heur function dict is a Nonetype by default.
        self.valid_ptypes = [key for key,val in build_args.items() if val is not None]

        if len(self.valid_ptypes) != 1:
            raise Exception('Exactly one valid prompt type must have been configured! We do not support any cross-interactions '
            'between prompt types in the prototype pseudo-mixture model.')
        
        if 'scribbles' in self.valid_ptypes or 'bboxes' in self.valid_ptypes or 'lassos' in self.valid_ptypes:
            raise NotImplementedError('We have selected bbox, scribbles, or lassos in the prompt gen. configs but they are ' \
            'not implemented in the utilities currently.')
        
              
    def check_toggle_dict(self):
        '''
        Function which checks that the configs provided for the heuristic prompting is supported or that the configs are
        cross-compatible. 
        '''
        #First check that only the point prompt type is invoked.
        if any(pt != 'points' for pt in self.valid_ptypes):
            raise ValueError('The basic random agent prompter only supports point-based prompting. \n'
                             'Please ensure only point-based prompting is configured.')
        #Next checking that only the valid prompt types have heuristic functions provided. 

        for prompt_type in self.supported_prompts:
            if prompt_type in self.valid_ptypes:
                if self.toggling_dict['intra_heur_level'].get(prompt_type) == None:
                    raise ValueError(f'The basic random agent prompter requires a prompter. \n'
                                     f'Please provide a heuristic function for prompt type: {prompt_type}.')
                if self.toggling_dict['heur_fn_dict'].get(prompt_type) == None:
                    raise ValueError(f'The basic random agent prompter requires a heuristic function to be provided. \n'
                                     f'Please provide a heuristic function for prompt type: {prompt_type}.')
                for heur_name, heur_fn in self.toggling_dict['heur_fn_dict'][prompt_type].items():
                    if not callable(heur_fn):
                        raise ValueError(f'The basic random agent prompter requires a callable heuristic function to be provided. \n'
                                         f'Please provide a callable heuristic function for prompt type: {prompt_type}.')
                    if heur_name not in self.toggling_dict['intra_heur_level'][prompt_type]:
                        raise ValueError(f'The basic random agent prompter requires that the heuristic function names match the heuristic param names provided. \n'
                                         f'Please ensure the heuristic function name: {heur_name} matches a heuristic param name for prompt type: {prompt_type}.')
            else:
                if self.toggling_dict['intra_heur_level'].get(prompt_type) != None:
                    raise ValueError(f'The basic random agent prompter requires was provided a heuristic function for a non-valid prompt type. \n'
                                     f'Please remove the heuristic function for prompt type: {prompt_type}.')
                if self.toggling_dict['heur_fn_dict'].get(prompt_type) != None:
                    raise ValueError(f'The basic random agent prompter requires was provided a heuristic function for a non-valid prompt type. \n'
                                     f'Please remove the heuristic function for prompt type: {prompt_type}.')
            #Next check that the heuristic function has been provided a heuristic param dict. 
        
    def init_toggle_dict(self,
                        heur_fn_dict: dict,
                        heur_build_args: dict,
                        mixture_args: Union[dict, None]):
        
        if heur_build_args is None:
            raise Exception('Heuristic level build arguments are ALWAYS required. At least one per heuristic!')
        elif heur_build_args is not None and mixture_args is None:
            #In this case, we will resort to defaults for the mixture methods.
            self.toggling_dict = {
                'class_level': None, 
                #None = Just use a default provided.
                'inter_prompt_level': None, 
                #None = Just use a default provided. 
                'intra_prompt_level': None,
                #None = Just use a default provided.
                'intra_heur_level': heur_build_args,
                #Use the heuristic build args provided.
                'heur_fn_dict': heur_fn_dict
            }
        else:
            raise NotImplementedError('Not implemented anything for handling the toggling of anything non-default wrt mixture strategies.')
        
    def filter_empty(self,
            prompts:Union[list[torch.Tensor], None], 
            prompts_lbs: Union[list[torch.Tensor], None] 
            ):
        '''
        Function which performs checks on the prompts and prompts labels to ensure that they match one another, and
        then evaluates how to filter the values according to what the prompts and prompt labels contain (or lack-thereof)
        '''
        if not prompts and not prompts_lbs: #In instances where list is empty, or where it is NoneType both eval as False.
            prompts = None
            prompts_lbs = None 
        elif bool(prompts) ^ bool(prompts_lbs): #Logical XOR, if they do not match then there is an error!
            raise Exception('One of the prompts and prompt labels evaluated as a False, whereas the other was True. Mismatch.')
        elif bool(prompts) & bool(prompts_lbs): #If both are true-types that are not empty.
            if len(prompts) != len(prompts_lbs):
                raise Exception('Non-matching quantity of prompts and prompts labels')
            else:
                pass #Made explicit for readability.
        
        return prompts, prompts_lbs 
            
    def filter_empty_dict(self, 
            prompts_dict: dict[str, dict[str, Union[list[torch.Tensor], None]]], 
            prompts_lbs_dict: dict[str, dict[str, Union[list[torch.Tensor], None]]], 
            ):
        '''
        Function which filters out the empty prompt types in output dicts into NoneTypes.

        inputs:

        prompts: A dictionary, separated by prompt types containing the lists of prompt inputs.
        '''
        for batch_idx in prompts_dict.keys():
            prompts_subdict = prompts_dict[batch_idx]
            prompts_lbs_subdict = prompts_lbs_dict[batch_idx]

            if not {f'{i}_labels' for i in set(prompts_subdict)} == set(prompts_lbs_subdict):
                raise Exception('The prompts dict and the prompts labels dict somehow did not contain the same prompt types')
                    
            for prompt_type in prompts_subdict.keys():
                p_list = prompts_subdict[prompt_type]
                p_lbs_list = prompts_lbs_subdict[f'{prompt_type}_labels']

                p_updated, p_lbs_updated = self.filter_empty(p_list, p_lbs_list)
                
                prompts_subdict.update({prompt_type: p_updated})
                prompts_lbs_subdict.update({f'{prompt_type}_labels': p_lbs_updated})

            prompts_dict[batch_idx] = prompts_subdict
            prompts_lbs_dict[batch_idx] = prompts_lbs_subdict

        return prompts_dict, prompts_lbs_dict
    
    def discrete_checker(self, data: Union[list[torch.Tensor], None]):
        '''
        Function which checks and converts input data which is containing discrete values (e.g. perhaps some prompts, 
        labels), to ensure that they are torch.int datatypes. Also moves it to the device specified.

        inputs: 
        
        data: An optional list of torch tensors (size is not relevant).
        If NoneType then ignores. 

        returns: 

        data: A list of torch tensors but in the correct datatype (torch.int) or NoneType for unused instances.
        '''

        if not data: #If empty list, or NoneType, just pass through.
            pass #Explicit logic provided here.... for debugging.
        else:
            for idx, tensor in enumerate(data):
                if tensor.is_complex():
                    raise Exception('No complex numbers should be possible.')
                
                #NOTE: Removed the floating point 
                elif tensor.is_floating_point():
                    data[idx] = tensor.to(dtype=torch.int32) #Just use int32, it is safe as the data is typically small
                    #anyways. 
        return data    
    
    def device_processor(self, data: Union[list[torch.Tensor], None], device: torch.device):
        '''
        Function which checks and/or moves it to the device specified.

        inputs: 
        data: An optional list of torch tensors (size is not relevant).
        If NoneType then ignores. 

        returns: 
        data: A list of torch tensors but on the correct device/ or NoneType for unused instances.
        '''
        if not data: #If empty list, or NoneType, just pass through.
            pass #Explicit logic provided here.... for debugging.
        else:
            for idx, tensor in enumerate(data):
                if tensor.device != device:
                    data[idx] = tensor.to(device=device)
        return data
        
    def output_processor(self, prompts_dict: dict, prompts_lbs_dict: dict):
        '''
        Function which post-processes the prompts dictionary and prompts labels dictionary to ensure they are in 
        the correct format, and on the correct device.
        '''
        
        #We first run it through a function which filters empty lists, checks for any inconsistencies between 
        #prompts and labels across all prompt types. 
        prompts_dict, prompts_lbs_dict = self.filter_empty_dict(prompts_dict=prompts_dict, prompts_lbs_dict=prompts_lbs_dict)

        #We then perform any datatype processing
        for batch_idx in prompts_dict.keys():
            prompts_subdict = prompts_dict[batch_idx]
            prompt_lbs_subdict = prompts_lbs_dict[batch_idx]
            for ptype, p_lb_type in zip(prompts_subdict.keys(), prompt_lbs_subdict.keys()):
                tmp_plist = copy.deepcopy(prompts_subdict[ptype])
                tmp_plb_list = copy.deepcopy(prompt_lbs_subdict[f'{ptype}_labels'])

                if ptype in self.discrete_variables:
                    tmp_plist = self.discrete_checker(data=tmp_plist)
                if p_lb_type in self.discrete_variables:
                    tmp_plb_list = self.discrete_checker(data=tmp_plb_list)

                prompts_subdict[ptype] = tmp_plist 
                prompt_lbs_subdict[p_lb_type] = tmp_plb_list

            prompts_dict[batch_idx] = prompts_subdict
            prompts_lbs_dict[batch_idx] = prompt_lbs_subdict
        return prompts_dict, prompts_lbs_dict
    
    def init_prompts(self, batch_size):
        #Initialises any required prompt structures. 
        prompts = dict()
        prompts_lbs = dict()
        for batch_idx in range(batch_size):
            prompts[batch_idx] = {
                'points': [],
                'scribbles': [],
                'bboxes': [],
                'lassos': []
            }
            prompts_lbs[batch_idx] = {
                'points_labels': [],
                'scribbles_labels': [],
                'bboxes_labels': [],
                'lassos_labels': []
            }
        return prompts, prompts_lbs

    def init_sample_regions_no_components(self, gt: torch.Tensor, pred: Union[torch.Tensor, None]):
        #Binary masks: BHWD, preds: BHWD torch tensors. 
        '''
        Very basic function which initialises the prompt sampling regions without a per-component basis. 
        Returns a dict of  regions for gt split by class, and also the same for the error regions.

        It is not intended for sophisticated behaviours, e.g. where one must consider the per-component basis, or the
        scale of a class. 

        For instances such as initialisation, the sampling region is just the gt.

        inputs: 
        
        pred - Optional (torch.Tensor) discretised pred (not one-hot encoded)
        gt - torch.Tensor which is discrete and not one-hot encoded.

        returns:
        
        sampling_regions_dict: 
        This dictionary chunks up the false negative error region in the following manner: It splits it by the 
        ground truth class of the error voxels.

        NOTE: For any instances where a gt class is empty an error would be raised, this is because we are only using binary semantic seg 
        formulations for now!

        Or, if the error regions are all totally empty then an error is raised because this should have been handled upstream in the iterative loop.
        '''
        batch_size = gt.shape[0]
        if pred is not None:
            assert batch_size == pred.shape[0]
            assert gt.shape == pred.shape 

        sampling_regions_dict = dict.fromkeys(self.semantic_id_dict.keys(), None)

        if not isinstance(gt, torch.Tensor) and not isinstance(gt, MetaTensor):
            raise Exception('The ground truth must be a torch tensor or a metatensor.')
        #Place gt on device and in int8 dtype. 

        if not gt.device == self.sim_device:
            gt = gt.to(dtype=torch.int8, device=self.sim_device)
        
        #NOTE: We will now allow for this to happen. it is unreasonable to assume that the network should not
        #observe fully empty patches at any point during inference, and it further complicates validation on 
        #small dataset sizes by limiting statistical power. 
        
        # #We will check that the gt isn't fully empty! 
        # if not gt.sum():
        #     raise Exception('The ground truth cannot be fully empty when prompting with this basic prompter.')

        if pred == None:
            #In this case, we only have the gt to work with, no prior pred.
            #We then split the gt, by class for each class. 
            accum = None
            for label, value  in self.semantic_id_dict.items():
                #We split gt by label. 
                if not (gt == value).sum(): #0 evaluates to bool False.
                    sampling_regions_dict[label] = None #It may be the case that we get a fully single class patch! 
                    #Could also just pass...
                else:
                    sampling_regions_dict[label] = gt == value
                
                    if accum is None:
                        accum = sampling_regions_dict[label]
                    else:
                        #We cumulatively add the gt regions, checking for overlaps with the current region under consideration.
                        if (accum & sampling_regions_dict[label]).sum():
                            raise Exception(f'Overlap detected between gt regions') 
                        #If it passes the overlap check, we add to the accum for the next region check.
                        accum = accum | sampling_regions_dict[label]

            accum = accum.to(device='cpu')
            torch.cuda.empty_cache()
            del accum

            if all([masks is None for label, masks in sampling_regions_dict.items()]): # if label.title() != 'Background']):
                raise Exception('All of the classes in the ground truth cannot be empty, we need something to sample from.')
            
            for batch_idx in range(batch_size):
                #Now we check on a batch level if the classes are all empty for any batch index.
                if not any([sampling_regions_dict[label][batch_idx].sum() > 0 
                            for label in sampling_regions_dict.keys() if sampling_regions_dict[label] is not None]): #or label.title() != 'Background']):
                    #We can skip over the none regions because they arealready empty, and so any() will flag an empty list as Falsey.
                    raise Exception(f'All of the error sampling regions are empty for batch index {batch_idx}.')
                
            gt = gt.to(device='cpu')
            del pred, gt
            torch.cuda.empty_cache()
        else:
            #In this case, we have a pred provided, so we can compute the error regions. 
            #  
            #Place pred on device and in int8 dtype (not bool yet!). 

            if not pred.device == self.sim_device:
                pred = pred.to(dtype=torch.int8, device=self.sim_device)
            
            #Find the false negative error region. 
            error_map_bool = pred != gt #We want it in bool format to minimise memory usage! Same memory usage as int8 though.
            
            accum = None
            for l1, v1 in self.semantic_id_dict.items():
                if not (error_map_bool & (gt == v1)).sum():
                    sampling_regions_dict[l1] = None 
                else: 
                    # err_regions_dict[l1] = split_by_gt
                    sampling_regions_dict[l1] = error_map_bool & (gt == v1) 
                    if accum is None:
                        accum = sampling_regions_dict[l1]
                    else:
                        #We cumulatively add the error regions, checking for overlaps with the current region under consideration.
                        if (accum & sampling_regions_dict[l1]).sum():
                            raise Exception(f'Overlap detected between error regions') 
                        #If it passes the overlap check, we add to the accum for the next region check.
                        accum = accum | sampling_regions_dict[l1]

            #First sanity check:
            if accum is None:
                raise Exception('The error regions cannot be fully empty for all samples, should have terminated the iterative loop already..')
            
            #Deep checks now, we will be doing a batch level check to ensure that no class is fully empty also.

            #Checking the sampling regions:
            if all([masks is None for masks in sampling_regions_dict.values()]):
                raise Exception('All of the class-separated sampling regions cannot be empty, should have terminated the iterative loop already..')

            #Now we will do a check on a batch level, to see if sampling regions are empty on a batch level. In which case we wouldn't be able to 
            #sample prompts for those batch samples! 

            #All of the classes cannot be empty on a batch level! 
            for batch_idx in range(batch_size):
                if not any([sampling_regions_dict[label][batch_idx].sum() > 0 
                            for label in sampling_regions_dict.keys() if sampling_regions_dict[label] is not None]):
                    #We can skip over the none regions because they arealready empty, and so any() will flag an empty list as Falsey.
                    raise Exception(f'All of the error sampling regions are empty for batch index {batch_idx}.')
                
            #Dumping VRAM as quickly as possible. 
            accum = accum.to(device='cpu')
            error_map_bool = error_map_bool.to(device='cpu')
            del accum, error_map_bool
            torch.cuda.empty_cache()

            #We no longer need the gt and the pred. Lets dump VRAM as quickly as possible. 
            pred = pred.to(device='cpu')
            gt = gt.to(device='cpu')
            del pred, gt
            torch.cuda.empty_cache()  
        return sampling_regions_dict
        
    def generate_prompts(
        self, 
        sampling_regions_dict: dict,
        tracked_prompts: dict,
        tracked_prompts_lbs: dict,
        mask_shape: torch.Size #This is the batchwise shape of the mask BHWD
        ):
        '''
        sampling _regions_dict: A dictionary containing the sampling regions for each class label (BHWD) mask. 
        tracked_prompts: A dictionary containing the tracked prompts for each batch index and prompt type.
        tracked_prompts_lbs: A dictionary containing the tracked prompts labels for each batch index and prompt type.
        '''
        #For now, we will use a vectorised implementation for sample generation, which strips out a lot of the layers
        #in sampling prompts in a complex manner in favour of a more direct approach. We have nonetheless left the structure
        #in place for potential future work. 
        assert self.toggling_dict['class_level'] is None, 'Class level mixture strategies not supported in basic random agent prompter.'
        assert self.toggling_dict['inter_prompt_level'] is None, 'Inter prompt level mixture strategies not supported in basic random agent prompter.'
        assert self.toggling_dict['intra_prompt_level'] is None, 'Intra prompt level mixture strategies not supported in basic random agent prompter.'
        assert self.toggling_dict['intra_heur_level'] is not None, 'Intra heuristic level params MUST be provided in basic random agent prompter.'
        #We will now generate the prompts according to the heuristic function provided.
        #We will also assert that only one heuristic function is provided per prompt as thereis no intra-prompt level 
        #support. 
        if any([len(self.toggling_dict['intra_heur_level'][ptype]) > 1 for ptype in self.supported_prompts if self.toggling_dict['intra_heur_level'][ptype] is not None]):
            raise Exception('The basic random agent prompter does not support multiple heuristic functions per prompt type currently')
        #NOTE: If > 1 then we need to integrate the binary mask update! 
        
        #First we invert the dictionary, and then use that to generate a CBHWD tensor so that we can vectorise
        #prompt generation as much as possible. 
        inverted_dict = dict(zip(self.semantic_id_dict.values(), self.semantic_id_dict.keys()))
        #Here we will completely bypass any looping through class, inter-prompt and intra-prompt levels as they are not supported.
        #and so we will directly pass through to the heuristic function level.
        for prompt_type in self.valid_ptypes:
            if self.toggling_dict['intra_heur_level'][prompt_type] != None:
                #Now we prompt. We will use basic logic here which is to prompt every class with the given heuristic.
                #Is this aligned with how they originally trained this? Not really, they assume each interaction is a 
                #separate event but this requires more logic to identify the compatible regions for prompting which also
                #slows down prompt generation.
                for heur, heur_fn in self.toggling_dict['heur_fn_dict'][prompt_type].items():
                    heur_params = self.toggling_dict['intra_heur_level'][prompt_type][heur]
                    if heur_params is None:
                        raise Exception(f'Heuristic parameters must be provided for heuristic function {heur} under prompt type {prompt_type} in basic random agent prompter.')
                    #Generate the prompts.

            
                    #This strategy ensures that we structure the classes in the same order as the numeric values in the configs 
                    #labels dict so that the prompt labels can be outputted correspondingly without needing to abuse the 
                    #for loop formulation of iterating through classes.
                    new_prompts, new_prompts_lbs = heur_fn(
                        torch.stack(
                            [
                            sampling_regions_dict[inverted_dict[k]] if sampling_regions_dict[inverted_dict[k]] != None else torch.zeros(mask_shape, dtype=torch.bool, device=self.sim_device)
                            for k in range(len(inverted_dict))
                            ], 
                        dim=0), 
                        heur_params)
                    #Extend the lists in the tracked prompts dicts.
                    for batch_idx in range(mask_shape[0]):
                        tracked_prompts[batch_idx][prompt_type] += new_prompts[batch_idx]   
                        tracked_prompts_lbs[batch_idx][f'{prompt_type}_labels'] += new_prompts_lbs[batch_idx]
                

            else:
                continue #No heuristic function provided for this prompt type, skip.

        return tracked_prompts, tracked_prompts_lbs

    def update_error_region(self, region_mask, prompts: list[torch.Tensor], prompt_type: str):
        '''
        This is a function which updates a region mask according to a set of prompts.

        This can be incorporated into an approach for multi-component handling, multi-class handling and also for 
        handling different prompt types.

        In particular, it can handle points and scribbles under the umbrella of free-form prompts. And box/lasso regions
        under the umbrella of region-based prompts. It assumes that the prompts are provided as a list of
        tensors with shape N x N_dim (N = 1 for points, and N_s.p for scribbles). It will raise an exception if
        the spatial dimensions are larger than N_dims. 

        inputs:

        region_mask: A binary mask with N_dim spatial dims denoting an error region with values of 1, everywhere else is zero. 
        prompts: A list of prompts N x N_dim for updating the region mask.
        prompt_type: The name of the prompt type: points, scribbles, bboxes, lassos.
        '''
        raise NotImplementedError('Not yet updated to work with batchwise masks and prompts')

        if prompts == []:
            return region_mask

        if prompt_type in self.free_form_prompts_ls:
            #Instead, we will fuse all of the coordinates, hence permitting for the handling to occur in a single step after merging.
            coords = torch.cat(prompts, dim=0)
            if not coords.shape[1] == region_mask.dim():
                raise Exception('The spatial dimensions of the input prompts must match the number of spatial dimensions in the mask.')

            if region_mask.device != self.sim_device:
                region_mask.to(device=self.sim_device)

            # for coords in free_form_prompts:  
            region_mask = update_binary_mask_freeform(coords, region_mask)

        elif prompt_type in self.partition_prompts_ls:
            raise NotImplementedError('Partition based prompt region updating not implemented yet.') 
        else:
            raise Exception(f'Prompt subtype {prompt_type} not recognised for region updating.')
        return region_mask
    
    
    def __call__(self, data):
        '''
        Function which calls on the methods for implementing the prompt generation process. 

        inputs: 

        data: A dictionary containing the following relevant fields: 

        gt: Torch tensor OR Metatensor containing the ground truth map.
        prev_pred (NOTE: OPTIONAL, is NONE otherwise) output pred from the inference call of the prior iteration.
       
        Two relevant fields for prompt generation contained are the: 
            pred: A tensor:
                1) (BHW(D)) containing the previous segmentation/prediction.
            
        im: Optional (or NoneType) dictionary containing the interaction memory from the prior interaction states.      
        '''

        if self.use_mem:
            #Extract the interaction memory.
            im = data.get('im')

            if not im:
                raise Exception('If using interaction memory, then it requires interaction memory available! Received nonetype')
            
            raise NotImplementedError('Not permitting the use of interaction memory in the prototype prompt generator, no memory conditioning.') 
        else:
            if data.get('prev_pred') == None:
                init_bool = True
            else:
                
                # gt = data['gt'][0,:].to(dtype=torch.int8, device=self.sim_device)
                if not isinstance(data['prev_pred'], torch.Tensor) and not isinstance(data['prev_pred'], MetaTensor):
                    raise TypeError('The pred needs to be a torch tensor or a Monai MetaTensor')            
                init_bool = False
            
            if not isinstance(data['gt'], MetaTensor) and not isinstance(data['gt'], torch.Tensor):
                raise TypeError('The gt needs to be a Monai MetaTensor or a torch Tensor')
            
            #Extracts the sampling regions.
            sampling_regions_dict = self.init_sample_regions_no_components(
                pred=data['prev_pred'].to(dtype=torch.int8, device=self.sim_device) if not init_bool else None,
                #Loading the gt.
                gt = data['gt'].to(dtype=torch.int8, device=self.sim_device)
        )
            #To prevent VRAM segfault for huge images just in case anything is lingering.
            torch.cuda.empty_cache() 

            #We initialise the prompt dictionaries on each call.
            tracked_prompts, tracked_prompts_lbs = self.init_prompts(batch_size=data['gt'].shape[0])

            #Now we generate the prompts, extremely basic sampling is only supported for our MVP.
            tracked_prompts, tracked_prompts_lbs = self.generate_prompts(
                sampling_regions_dict=sampling_regions_dict,
                tracked_prompts=tracked_prompts,
                tracked_prompts_lbs=tracked_prompts_lbs,
                mask_shape=data['gt'].shape #Including the batch dimension! This is a per-class BHWD shape.
            )
            tracked_prompts, tracked_prompts_lbs = self.output_processor(tracked_prompts, tracked_prompts_lbs)
            return tracked_prompts, tracked_prompts_lbs
        
prompt_mixture_registry = {
    'RandomAgent': make_factory(RandomAgent)
}