#TODO: Needs to be updated so that it is compatible with use cases with batch size > 1. 

# import os
# from os.path import dirname as up
# import sys
# sys.path.append(up(up(os.path.abspath(__file__))))
# from typing import Union 
# import torch
# from nnInteractive.adaptation.training_utils.prompters.mixture_registry import prompt_mixture_registry
# from nnInteractive.adaptation.training_utils.prompters.heuristic_registry import base_prompter_registry

# class BuildHeuristic:

#     def __init__(self,
#                 sim_device: torch.device,
#                 use_mem:bool,
#                 config_labels_dict:dict,
#                 heuristics:dict,
#                 heuristic_params: dict,
#                 heuristic_mixtures: Union[dict, None],
#                 heuristic_class_type: str
#                 ):
        
#         '''
#         Heuristic Prompt Generation builder. This class makes use of the simulation arguments, and constructs a simulation
#         class which can be called for generation of prompts and labels in the list[torch] format. 

#         Inputs:

#         sim_device: torch.device - The device which the computations will be implemented on for gpu (or cpu) processing.

#         use_mem: A bool - Denotes whether the interaction memory dictionary will be used so that stored memory
#         is being retained/used to filter the error regions for prompt generation.

#         config_labels_dict: A dictionary mapping the class labels to the class integer codes. 

        
        
#         (REQUIRED) heuristics: Dict - Simulation methods used for generating each prompt (currently: [points, scribbles, bbox]). 
        
#         Must always provide for all prompt types. A dictionary first separated by the prompt type:

#         Within each prompt type, containing the list of prompting strategies being used for prompt simulation. 
#         This dictionary allows the flexibility to permit combinations/lists of strategies. 

#         NOTE: Instances where a specific prompt type is never being used, will require that the value be a NoneType 
#         arg instead. The prompts generated will also be a NoneType argument for the given prompts also! 

#         NOTE: If any prompt drop-out methods are used, it is possible for an output to be a NoneType even if the 
#         input arg for the prompt type is not.

        
        
#         (REQUIRED) heuristic_params: A twice nested dictionary, for each prompt strategy within a prompt type it 
#         contains a dictionary of build arguments for each corresponding strategy implemented. 

#         Similar structure as prompt_methods, as they must correspond together!:

#         NOTE: For each heuristic it must contain information about handling at the heuristic-level.

#         NOTE: For prompt types which are not valid, it should be a NoneType. 
        
#         NOTE: CAN NEVER BE NONETYPE OTHERWISE! 


#         (OPTIONAL)heuristic_mixture: An arbitrarily nested dict denoting a strategy for the cascading functions which
#         outline the basic strategy of handling prompting interactions, e.g. class-level toggling, inter-prompt level toggling, 
#         intra-prompt level toggling.

#         This heuristic mixture arg will control whether/how prompt-methods will interact/condition one another 
#         during the simulation (e.g. constraining error regions, prompt placements, or even switching on/off specific 
#         prompts).


#         For example, a structure may look like: 
        
#         dict(
#             'class_level': dict of mixture args/None
        
#             'inter_prompt':dict[tuple(prompt cross-interaction combinations), mixture_args/None], 
                            
#             'intra_prompt': dict[prompt_type_str : dict[tuple(prompt_strategy cross-interaction combinations), mixture_args/None])
        
#             NOTE: Tuples can provide an immutable set of combinations

#             NOTE: Downstream use will likely necessitate the use of set logic to verify combinations as the tuple 
#             is immutable. Verification likely will entail the following: Generate set of potential combinations from
#             prompt_types (or strategy), cross-reference with the corresponding dict item by converting key into set.


#         NOTE: Can optionally be fully Nonetype i.e. default behaviour.


#         REQUIRED (heuristic class_type): The heuristic class type to be used for prompt generation. This provides some basic 
#         structure which acts as a skeleton to be filled out with the prior arguments when generating the prompts. 

#         '''
        
#         self.sim_device = sim_device 
#         self.use_mem = use_mem 
#         self.config_labels_dict = config_labels_dict
#         self.heuristics = heuristics
#         self.heuristic_params = heuristic_params
#         self.heuristic_mixtures = heuristic_mixtures 
#         self.heuristic_class_type = heuristic_class_type
#         self.free_form_prompts = ['points', 'scribbles']
#         self.partition_prompts = ['bboxes', 'lassos'] 

#         self.heuristic_caller = self.initialise_heuristics()

#         #Checking that at least one prompt is being used:

#         if not any(list(self.heuristics.values())):
#             raise ValueError('There must be at least one prompt type which is not a NoneType for simulation.')

#     def at_least_one_prompt(self, 
#                             generated_prompts:dict, 
#                             generated_prompts_labels:dict, 
#                             data: dict):
#         '''
#         This function will check whether at least one prompt type has a prompt, otherwise it will raise an exception.

#         Requirement: Any empty list should have been converted to a NoneType. We will check for this! 

#         inputs: 
        
#         generated_prompts: A prompt type separated dict containing the list of prompts.
#         generated_prompts_labels: A prompt type separated dict containing the list of prompts corresponding labels.
#         data: The data dictionary which was passed into the call operation, contains info about the prev_output_data.
#         '''

#         #Determining what the mode is implicitly from the data dictionary.

#         if data['prev_output_data'] is None:
#             mode = 'init'
#         else:
#             mode = 'edit' 

#         #Checking for empty lists to raise an exception about code elsewhere.

#         if [] in generated_prompts.values() or [] in generated_prompts_labels.values():
#             raise ValueError('Any empty prompt lists or prompt labels lists must be replaced with a NoneType for the value')


#         #If mode is init, then check it has at least one valid prompt type populating it

#         if mode == 'init':
#         # I.e., if it went through the exception handling for every valid prompt type then it would be only NoneTypes throughout.
#             if not any(generated_prompts.values()) or not any(generated_prompts_labels.values()):
#                 raise ValueError('There must be at least one prompt!')
#         elif mode == 'edit':
#             if not any(generated_prompts.values()) or not any(generated_prompts_labels.values()):
#                 raise ValueError('There must be at least one prompt!')
#             # if not any([generated_prompts[key] for key in self.free_form_prompts]) or not any([generated_prompts_labels[f'{key}_labels'] for key in self.free_form_prompts]):
#                 # raise ValueError('There must be one free_form prompt in the editing iterations.')
#             #NOTE: We have deprecated the above code, as it is not necessarily true that partition-based prompts (what we 
#             # previously were classifying as grounded prompts incorrectly) cannot be used for editing.

#         else:
#             raise Exception('Error in the implementation here.')
         

        
        
#     def initialise_heuristics(self):
#         '''
#         This function will initialise the heuristic call to be used, such that it can be called for prompt 
#         generation. It takes the abstract heuristics from the registry, and places them into a nested dictionary 
#         split by prompt type and then split by heuristic type.
        
#         Returns:
        
#         heur_caller: An initialised class which can be used to generate the prompts.

#         #TODO: Implementation for non-prototype mixtures incorporation.
#         '''

#         if not self.heuristic_mixtures:
#             #Here we will initialise the heuristics for prototype prompt generation method (prototype pseudo-mixture).
#             heur_fn_dict = dict()

#             for prompt_type, heuristics in self.heuristics.items():
#                 prompt_heur_fns = dict() 
                
#                 if heuristics: #If ptype heuristics is not a NoneType or empty/I.e. if config exists for a prompt type
#                     if self.heuristic_params[prompt_type] is None:
#                         raise Exception('There must be args provided at heuristic level, if heuristics are being provided')
#                     for heuristic in heuristics: 
#                         prompt_heur_fns[heuristic] = base_prompter_registry[prompt_type][heuristic]
#                         if self.heuristic_params[prompt_type][heuristic] is None: 
#                             raise Exception('There must be an arg provided at heuristic level, cannot be NoneType')

#                 else:
#                     prompt_heur_fns = None 
                
#                 heur_fn_dict[prompt_type] = prompt_heur_fns

#             # return mixture_class_registry[self.heuristic_class_type](
#             #     config_labels_dict=self.config_labels_dict,
#             #     sim_device=self.sim_device,
#             #     use_mem=self.use_mem,
#             #     build_args=self.heuristic_params,
#             #     mixture_args=self.heuristic_mixtures,
#             #     heur_fn_dict=heur_fn_dict,                                
#             #     )
#             return prompt_mixture_registry[self.heuristic_class_type](
#                 args={
#                     'config_labels_dict': self.config_labels_dict,
#                     'sim_device': self.sim_device,
#                     'use_mem': self.use_mem,
#                     'build_args': self.heuristic_params,
#                     'mixture_args': self.heuristic_mixtures,
#                     'heur_fn_dict': heur_fn_dict,                                
#                 })
#         else:
#             raise NotImplementedError('Implement the code for initialising non-prototype prompt mixture methods. I.e., \n' \
#             'those for which heuristic_mixtures is not a NoneType.')
#             '''
#             Info: 

#             self.heuristic_mixtures: A twice nested dict denoting a strategy for mixing prompt simulation across intra
#             and inter-prompting strategies. The mixture args are a dictionary for each corresponding pairing of cross-interactions.

#             This prompt mixture arg will control whether/how prompt-methods will interact/condition one another during the 
#             simulation. 

#                 Has structure: dict('inter_prompt':dict[tuple(prompt cross-interaction combinations), mixture_args/None], 
#                                     'intra_prompt': dict[prompt_type_str : dict[tuple(prompt_strategy cross-interaction combinations), mixture_args/None])
                
#                 Tuple can provide an immutable set of combinations

#                 NOTE: Downstream use will likely necessitate the use of set logic to verify combinations as the tuple 
#                 is immutable. Verification likely will entail the following: Generate set of potential combinations from
#                 prompt_types (or methods), cross-reference with the corresponding item by converting key into set.
#             '''

#     def extract_prompts(self, data: dict):
#         '''
#         This function initialises then populates the generated prompts and labels using the heuristic caller.

#         Also calls on the output checker, to ensure that there is at least one prompt simulated.

#         Inputs:

#         data: This is a dictionary which contains the following information:
#             gt: Metatensor, or torch tensor containing the ground truth mask.
#             img: Metatensor, or torch tensor containing the image.

#             prev_output_data: (Optional) Dictionary containing the information from the prior inference calls 
#             OR NONETYPE (for init modes).

#             Contains the following fields:
#                 pred: A dict, which contains a subfield "metatensor", with a 1HWD Metatensor, or torch tensor 
#                 containing the discretised prediction mask from the prior inference call.
#                 probs: A dict, which contains a subfield "metatensor", with a CHWD Metatensor, or torch tensor 
#                 containing the probs map from the prior inference call.

#         '''

#         if not self.heuristic_mixtures:
#             #We populate the prompts:
#             generated_prompts, generated_prompt_labels = self.heuristic_caller(data)
                    
#             #We check that the output was generated correctly/with a valid output.
#             self.at_least_one_prompt(generated_prompts, generated_prompt_labels, data)

#             return generated_prompts, generated_prompt_labels
#         else: 
#             raise NotImplementedError('The heuristic mixture strategy has not yet been implemented')

#     def __call__(self, data):

#         '''
#         Inputs:

#         data: A dictionary containing the following fields: 

#         image: Torch tensor OR Metatensor containing the image in the native image domain (no pre-processing applied other than re-orientation in RAS)
#         gt: Torch tensor OR Metatensor containing the ground truth map in RAS orientation, but otherwise in the native image domain (no pre-processing other than RAS re-orientation).
        
#         prev_output_data: (NOTE: OPTIONAL, is NONE otherwise) output dictionary from the inference call which has been post-processed 
#         in the pseudo-ui front-end.
       
#         Two relevant fields for prompt generation contained are the: 
#             pred: A dictionary containing 3 subfields:
#                 1) "path": Path to the prediction file (Not Relevant)
#                 And two relevant subfields
#                 2) "metatensor" A Metatensor or torch tensor (1HW(D)) containing the previous segmentation in the native image domain (no pre-processing applied other than re-orientation in RAS) 
#                 3) "meta_dict" A dict containing (at least) the affine matrix for the image, containing native image domain relevant knowledge.
            
#             probs:
#                 1) "paths": List of paths to the prediction file (Not Relevant)
#                 And two potentially relevant subfields
#                 2) "metatensor" A Metatensor or torch tensor (CHW(D)) containing the previous segmentation in the native image domain (no pre-processing applied other than re-orientation in RAS) 
#                 3) "meta_dict" A dict containing (at least) the affine matrix for the image, containing native image domain relevant knowledge.
        
#         im: Optional dictionary containing the interaction memory from the prior interaction states.      

#         Returns: 

#         Both outputs are in the list[torch] format denoted in: `<https://github.com/IS_Validate/blob/main/src/data/interaction_state_construct.py>` 

#         prompts_torch_format: dict - A dictionary, separated by the prompt-type, which contains the prompt spatial information
#         for the selected prompt types in the prompt generation config.  

#         prompts_labels_torch_format: dict - A dictionary, separated by the prompt type, which contains the prompt
#         labels for the corresponding prompts (or NoneTypes for the empty prompts!). 
#         '''
#         if not self.heuristic_mixtures:
#             prompts_torch_format, prompts_labels_torch_format = self.extract_prompts(data)
        
#         elif self.heuristic_mixtures:
#             raise NotImplementedError('The heuristic mixture strategy has not yet been implemented')
    
#         return prompts_torch_format, prompts_labels_torch_format 