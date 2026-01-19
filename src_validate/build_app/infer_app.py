from __future__ import annotations
import os
import sys
import copy 
import warnings
import shutil
import json 
# import base64
app_local_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
pretrained_ckpt_dir = os.path.join(app_local_path, 'ckpt', 'nnInteractive_v1.0')
sys.path.append(app_local_path)
# from pathlib import Path
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnInteractive.adaptation.training_utils.saving_utils import save_checkpoint
from nnInteractive.inference.pretrained_inference_session import nnInteractiveInferenceSession as PreTrainednnInteractiveSession
from nnInteractive.inference.adapted_inference_session import nnInteractiveInferenceSession as AdaptednnInteractiveSession
from nnunetv2.utilities.helpers import empty_cache
from nnInteractive.adaptation.data_handling import DataHandler
from nnInteractive.adaptation.adaptation_trigger_criteria import AdaptationTriggerCriterionRegistry
from nnInteractive.adaptation.adaptation_planning import AdaptationPlanner
from nnInteractive.adaptation.adaptation_executor import AdaptationExecutor
adaptation_config_registry_path = os.path.join(app_local_path, 'nnInteractive', 'adaptation', 'adaptation_config_registry.txt') 
# adaptation_config_name = 'adapt_prototype_1'

meta_algo_state_keys = {
    'algo_cache_name',
    'adaptation_config',
    'dataset_info',
    'experiment_dir',
    'memory_buffer_dir',
    'memory_buffer_disk',
    'trigger_criterion_config',
    'data_split',
    'dataloader_transforms',
    'data_splitting_config',
    'unassigned_samples',
    'adaptation_plans',
    'app_params',
    'write_state',
    'adaptation_number',
    'configs_labels_dict'
}

training_state_keys = {
    'checkpoints',
    'best_ckpt_name',
}
class InferApp:
    def __init__(self, 
        infer_device, 
        adaptation_config_name: str, 
        algorithm_state: dict = {},
        enable_adaptation: bool = False,
        algo_cache_name: str = ''):

        self.infer_device = infer_device
        if self.infer_device.type != 'cuda':
            raise ValueError('This script only should be used with CUDA inference device.')

        self.algorithm_state = algorithm_state 
        self.adaptation_config_name = adaptation_config_name
        
        if self.algorithm_state == {} and not enable_adaptation:
            #In this case, there is no prior state to load nor are we adapting, we are just using the pretrained model. 
            #NOTE: This is just the zero-shot model.... so take that information into account.
            autoseg_infer = False #This is a variable for storing the action taken in the instance where there is no prompting information provided in a slice.
            #In the case where it is True, a prediction will be made, and the stored pred and output pred will be the same.
            #In the case where it is False, a prediction will not be made, the stored pred will be None, and the output pred will be zeroes.
            permitted_prompts = ('points', 'bboxes', 'scribbles', 'lasso') 
            prompt_subtypes = {
                'points':'free_prompts',
                'scribbles':'free_prompts', 
                'bboxes': 'partition_prompts',
                'lasso':'partition_prompts'
            }
            self.app_params = {
                'autoseg_infer_bool': autoseg_infer,
                'permitted_prompts': permitted_prompts,
                'prompt_subtypes': prompt_subtypes
            }
            self.dataset_info = None #No prior loaded dataset info for zero-shot inference. 
        elif (self.algorithm_state == {} or self.algorithm_state =={'meta_algorithm_state':{'algo_cache_name': algo_cache_name}}) and enable_adaptation:
            #In this case, we are going to be saving algorithm states. But there is no prior state to load 
            #or we early-exited before we could even save the initial checkpoint pre-evaluation.
            
            #Lets create some important info for the adaptation process.
            #In this case, we need to a create a new memory buffer which can persist in cases where the job is restarted.
            self.algo_cache_name = algo_cache_name
            experiment_subdir =  self.algo_cache_name
            experiment_dir = os.path.join(app_local_path, 'experiment_storage', experiment_subdir)
            memory_buffer_dir = os.path.join(experiment_dir, 'memory_buffer')
            if os.path.exists(memory_buffer_dir):
                shutil.rmtree(memory_buffer_dir)
                #We could be in a scenario where the memory buffer was created but the experiment was interrupted before
                # any checkpoint was saved. So we will erase and start-over. 
            os.makedirs(memory_buffer_dir) 
            #The experiment AND memory buffer directory should not exist yet if we are creating it now!

            #First loading the adaptation configuration from the registry. This is static for an experiment,
            #we will not be pausing and switching. Any adjustment needs to be handled by the adaptation
            #config/process itself.

            adaptation_config = self.extract_registry_config(
                path=adaptation_config_registry_path,
                name=self.adaptation_config_name
            )
            
            #Configuring some functionality parameters for initialisation.
            autoseg_infer = False #This is a variable for storing the action taken in the instance where there is no prompting information provided in a slice.
            #In the case where it is True, a prediction will be made, and the stored pred and output pred will be the same.
            #In the case where it is False, a prediction will not be made, the stored pred will be None, and the output pred will be zeroes.
            
            permitted_prompts = ('points', 'bboxes', 'scribbles', 'lasso') 
            prompt_subtypes = {
                'points':'free_prompts',
                'scribbles':'free_prompts', 
                'bboxes': 'partition_prompts',
                'lasso':'partition_prompts'
            }
            self.app_params = {
                'autoseg_infer_bool': autoseg_infer,
                'permitted_prompts': permitted_prompts,
                'prompt_subtypes': prompt_subtypes,
            }

            #Initialising some variables for tracking the adaptation process.
            self.adaptation_config = adaptation_config
            self.dataset_info = None #We will set this when we load the first sample for adaptation.
            self.experiment_dir = experiment_dir
            self.samples_memory_buffer_dir = memory_buffer_dir #A directory for storing the samples saved on disk for adaptation purposes.
            self.samples_memory_buffer_disk = dict() #A dictionary for storing the filepaths of the samples saved on disk for adaptation purposes.

            self.write_state = False #We only write the algorithm state after adaptation has been triggered. 
            

            #The next few parameters are usually only configured if adaptation has been triggered at least once.
            self.data_split = dict() #A dictionary for storing the data split info for adaptation purposes.
            self.unassigned_samples = [] #A list for storing samples which have been saved to disk but not yet
            # assigned to a data split.#samples, starts with Nonetype.
            self.dataloader_transforms = None #The dataloader transforms used for the last data split, should be None.
            # self.data_splitting_config = None 
            self.trigger_criterion_config = adaptation_config['adaptation_trigger_config']['criterion_config']
            self.data_splitting_config = adaptation_config['adaptation_trigger_config']['data_splitting_config']
            #This is typically the data splitting config used for the last data split (would only have an updated algo state if 
            #the checkpoint was being saved anyways..., so it should be a nonetype at this point)
            #NOTE: However, we currently assume this is a fixed parameter. 
            self.adaptation_plans = None 
            #May not always exist (i.e. be something other than a nonetype) if adaptation has never been triggered.
            # self.app_params = self.app_params #Current app_params before adaptation.
            self.adaptation_number = 0
            #Starting off with the zeroth adaptation iteration (i.e. none have happened)

            self.meta_algorithm_state = {
                'algo_cache_name': algo_cache_name,
                'adaptation_config': adaptation_config,
                'dataset_info': self.dataset_info,
                'experiment_dir': self.experiment_dir, 
                'memory_buffer_dir': self.samples_memory_buffer_dir,
                'memory_buffer_disk': self.samples_memory_buffer_disk,
                'trigger_criterion_config': self.trigger_criterion_config,
                'data_split': self.data_split,
                'dataloader_transforms': self.dataloader_transforms,
                'data_splitting_config': self.data_splitting_config,
                'unassigned_samples': self.unassigned_samples,
                'adaptation_plans': self.adaptation_plans,
                'adaptation_number': self.adaptation_number,
                'app_params': self.app_params,
                'write_state': self.write_state
                }
            
            # First the data handling class. 
            #for saving samples to disk for adaptation purposes and handling the data splitting.

            # There is no guarantee that the run will not beinterrupted nor can we store it all in memory, 
            # so we need to save to disk for automatic recovery!
            self.data_handler = DataHandler(
                dataset_constructor_config=adaptation_config['data_handling_config']['dataset_constructor_config'],
                io_operations_config=adaptation_config['data_handling_config']['io_operations_config'],
                dataloaders_transforms=self.dataloader_transforms
            )
            # Now we will need to initialise the classes for executing the adaptation itself. Planner config is FIXED
            #across an experiment! It is the entire thing underpinning this operation we can't just change this config
            #!!!
            self.adaptation_planner = AdaptationPlanner(
                planner_name=adaptation_config['adaptation_planner']['planner_registry_name'],
                planner_config=adaptation_config['adaptation_planner']['planner_config']
            )
            self.adaptation_trigger_criterion = AdaptationTriggerCriterionRegistry(
                criterion_config=self.trigger_criterion_config,
                data_splitting_config=self.data_splitting_config
                #We put this data splitting config here, because it is needed to cross-check that the data splitting
                #has been satisfied when trying to check the adaptation triggering criteria.
            )
            self.adaptation_executor = AdaptationExecutor(
                training_tmp_dir=os.path.join(experiment_dir, 'adaptation_tmp_dir'),
                completion_dir=os.path.join(experiment_dir, 'adaptation_completion_dir'),
                device=self.infer_device,
                max_workers=1
            )


            self.checkpoints = None 
            self.best_ckpt_name = None 
            self.algorithm_training_state = {
                'checkpoints': self.checkpoints,
                'best_ckpt_name': self.best_ckpt_name
            }  #No prior adaptation has been done yet.

            #Here we check that we have all the required keys. 
            if [i for i in self.meta_algorithm_state.keys() if i not in meta_algo_state_keys]:
                raise Exception('The constructed meta algorithm state did not have the correct keys! Cannot proceed!')
            if [i for i in self.algorithm_training_state.keys() if i not in training_state_keys]:
                raise Exception('The constructed algorithm training state did not have the correct keys! Cannot proceed!')
            self.algorithm_state = {
                'meta_algorithm_state': self.meta_algorithm_state,
                'algorithm_training_state': self.algorithm_training_state
            }

        elif (self.algorithm_state != {} and self.algorithm_state != {'meta_algorithm_state':{'algo_cache_name': algo_cache_name}}) and enable_adaptation:
            #Algo state exists, is not empty, and we are adapting further.
            if not type(self.algorithm_state) == dict:
                raise TypeError('The provided algorithm state must be a dictionary if adaptation is to be enabled!')
            
            #In this case, we are loading from a prior saved algorithm state, and we will be adapting further.
            self.meta_algorithm_state = self.algorithm_state.get('meta_algorithm_state') 
            if self.meta_algorithm_state is None:
                raise Exception('An algorithm state was provided for loading, but it did not contain any meta-algorithm state information! Cannot proceed with loading!')
            

            self.algorithm_training_state = self.algorithm_state.get('algorithm_training_state')
            if self.algorithm_training_state is None:
                raise Exception('An algorithm state was provided for loading, but it did not contain any algorithm training state information! Cannot proceed with loading!') 
            #Checking all the items are present.
            if [i for i in self.meta_algorithm_state.keys() if i not in meta_algo_state_keys]:
                raise Exception('The provided meta algorithm state did not have the correct keys! Cannot proceed with loading!')
            if [i for i in self.algorithm_training_state.keys() if i not in training_state_keys]:
                raise Exception('The provided algorithm training state did not have the correct keys! Cannot proceed with loading!')        


            #Loading variables for continuing adaptation:
            self.algo_cache_name = self.meta_algorithm_state.get('algo_cache_name') 
            if self.algo_cache_name != algo_cache_name:
                raise Exception('The provided algorithm state cache name does not match the requested cache name for this experiment! Cannot proceed with loading!')
            
            #Loading the adaptation config:
            if self.meta_algorithm_state.get('adaptation_config') is None:
                raise Exception('The provided algorithm state did not contain an adaptation configuration! Cannot proceed with loading!')
            else:
                #Currently we assume that the adaptation config is fixed for an experiment (even if some of the 
                    #configs inside it may be dynamic in the future.)
                registry_adaptation_config = self.extract_registry_config(
                    path=adaptation_config_registry_path,
                    name=self.adaptation_config_name)
                if not all([self.meta_algorithm_state.get('adaptation_config')[i] == registry_adaptation_config[i] for i in registry_adaptation_config.keys()]):
                    raise Exception('The adaptation configuration in the provided/loaded algorithm state does not match the \n'
                                    'requested adaptation configuration for the given experiment! Cannot proceed with loading!')
                else:
                    self.adaptation_config = self.meta_algorithm_state.get('adaptation_config')
            
            #Loading variables which are required for all adaptation scenarios.
            self.trigger_criterion_config = self.meta_algorithm_state.get('trigger_criterion_config')
            if self.trigger_criterion_config is None:
                raise Exception('The provided algorithm state did not contain an adaptation trigger criterion configuration! Cannot proceed with loading!')

            self.dataset_info = self.meta_algorithm_state.get('dataset_info')
            if self.dataset_info is None:
                raise Exception('The provided algorithm state did not contain dataset information! Cannot proceed with loading!')
            self.experiment_dir = self.meta_algorithm_state.get('experiment_dir')
            if self.experiment_dir is None:
                raise Exception('The provided algorithm state did not contain an experiment directory! Cannot proceed with loading!')
            if os.path.exists(self.experiment_dir) == False:
                raise Exception('The experiment directory provided in the algorithm state does not exist on disk! Cannot proceed with loading!')
            self.samples_memory_buffer_dir = self.meta_algorithm_state.get('memory_buffer_dir')
            if self.samples_memory_buffer_dir is None or not self.samples_memory_buffer_dir:
                raise Exception('The provided algorithm state did not contain a memory buffer directory or was empty! Cannot proceed with loading from a \n'
                                'previous meta_algo state!')
            self.samples_memory_buffer_disk = self.meta_algorithm_state.get('memory_buffer_disk')
            if self.samples_memory_buffer_disk is None or not self.samples_memory_buffer_disk:
                raise Exception('The provided algorithm state did not contain a memory buffer disk dictionary or was empty! Cannot proceed with loading from a \n'
                                'previous meta_algo state!')
            if not os.path.exists(self.samples_memory_buffer_dir):
                raise Exception('The memory buffer directory provided in the algorithm state does not exist on disk! Cannot proceed with loading!')
            
            
            #Pulling some variables which can be dependent on whether adaptation has been triggered before or not.
            self.adaptation_number = self.meta_algorithm_state.get('adaptation_number')
            self.data_split = self.meta_algorithm_state.get('data_split')
            self.dataloader_transforms = self.meta_algorithm_state.get('dataloader_transforms')
            self.data_splitting_config = self.meta_algorithm_state.get('data_splitting_config')
            self.unassigned_samples = self.meta_algorithm_state.get('unassigned_samples')
            self.adaptation_plans = self.meta_algorithm_state.get('adaptation_plans')
            if self.adaptation_number > 0:
                # datasplit and transforms may be nonetype if no prior adaptation has been triggered.
                if self.data_split is None:
                    raise ValueError('The data split info in the provided algorithm state cannot be nonetype if adaptation number is greater than 0! Cannot proceed with loading!')
                else:
                    if not type(self.data_split) == dict:
                        raise TypeError('The data split info in the provided algorithm state must be a dictionary! Cannot proceed with loading!')
                    if not self.data_split: 
                        raise ValueError('The data split info in the provided algorithm state cannot be an empty dictionary if adaptation number is greater than 0! Cannot proceed with loading!')
                if self.dataloader_transforms == None:
                    raise ValueError('The dataloader transforms in the provided algorithm state cannot be nonetype if adaptation number is greater than 0! Cannot proceed with loading!')
                if self.data_splitting_config == None:
                    raise ValueError('The data splitting config in the provided algorithm state cannot be nonetype if adaptation number is greater than 0! Cannot proceed with loading!')
                else:
                    if not type(self.data_splitting_config) == dict:
                        raise TypeError('The data splitting config in the provided algorithm state must be a dictionary! Cannot proceed with loading!')
                if self.adaptation_plans == None:
                    raise ValueError('The adaptation plans in the provided algorithm state cannot be nonetype if adaptation number is greater than 0! Cannot proceed with loading!')
            else:
            #This may be a nonetype if no adaptation has been triggered yet
                if self.data_split is None:
                    raise ValueError('The data split info in the provided algorithm state must be a nonetype if adaptation number is 0! Cannot proceed with loading!')
                if self.data_split != {}:
                    raise ValueError('The data split info in the provided algorithm state must be an empty dictionary if adaptation number is 0! Cannot proceed with loading!')
                if self.dataloader_transforms is not None:
                    raise ValueError('The dataloader transforms in the provided algorithm state must be a nonetype if adaptation number is 0! Cannot proceed with loading!')
                if self.data_splitting_config == None:
                    raise ValueError('The data splitting config in the provided algorithm state currently cannot be a nonetype \n' 
                                     'Even if not adapted yet we need some configuration to start off with as it forms part of the trigger \n'
                                     'condition (i.e., insufficient samples to split -> cannot trigger). Cannot proceed with loading!')
                if self.adaptation_plans is not None:
                    raise ValueError('The adaptation plans in the provided algorithm state must be a nonetype if adaptation number is 0! Cannot proceed with loading!')
            
            #We will now load the existing app_params
            self.app_params = self.meta_algorithm_state.get('app_params') 
            
            #We will now load variables which are required for adaptation.
            self.checkpoints = self.algorithm_training_state.get('checkpoints')
            self.best_ckpt_name = self.algorithm_training_state.get('best_ckpt_name')
            #We will now check that these are valid.
            if not all([os.path.exists(path) for path in self.checkpoints.values()]):
                raise ValueError('One or more of the checkpoint paths stored in the provided algorithm training state do not exist on disk! Cannot proceed with loading!')
            if self.best_ckpt_name not in self.checkpoints.keys():
                raise ValueError('The best checkpoint name stored in the provided algorithm training state is not a valid key in the checkpoints dictionary! Cannot proceed with loading!')

            #Write state is false until adaptation has been triggered always! So lets set this to false again. 
            self.write_state = False

            #Here we will be in a scenario where we have an existing memory buffer. So we will need to just clear the ones
            #which weren't stored in the checkpoint memory buffer disk dictionary. Because the sample is written to 
            #disk before the checkpoints are updated (just due to the workflow.)

            self.clear_untracked_memory_buffer_samples(
                memory_buffer_dir=self.samples_memory_buffer_dir,
                tracked_memory_buffer_dict=self.samples_memory_buffer_disk
            )

            #Less likely, but still possible, is the scenario where there are unassigned samples which are not in the tracked memory buffer somehow.
            self.clear_untracked_unassigned_samples(
                unassigned_samples_list=self.unassigned_samples,
                tracked_memory_buffer_dict=self.samples_memory_buffer_disk
            )
            #NOTE: This does not clear the sample on the disk, but just from the unassigned samples list because
            #it was not yet written to a checkpoint, and so it means that any sample which was not on the buffer
            #needs to be re-evaluated. Unlikely to be triggered however, because the list would only be updated
            #if the checkpoint had been updated anyways... (we are just being extra careful). 

            #We will check whether any unassigned samples are here which are not in the tracked memory buffer.
            # If so, we will remove them from the unassigned samples list, as the re-run will go back over
            # these samples again in the evaluation pipeline.

            if self.checkpoints == None:
                raise ValueError('The checkpoints dictionary cannot be None when loading from a prior algorithm state! Cannot proceed with loading!')
            if self.best_ckpt_name == None:
                raise ValueError('The best checkpoint name cannot be None when loading from a prior algorithm state! Cannot proceed with loading!')

            #TODO: Initialise the classes here used for the adaptation process, similar to above.

            self.data_handler = DataHandler(
                dataset_constructor_config=self.adaptation_config['data_handling_config']['dataset_constructor_config'],
                io_operations_config=self.adaptation_config['data_handling_config']['io_operations_config'],
                dataloaders_transforms=self.dataloader_transforms
                )
            #Now we will need to initialise the classes for executing the adaptation itself.
            self.adaptation_planner = AdaptationPlanner(
                planner_name=self.adaptation_config['adaptation_planner']['planner_registry_name'],
                planner_config=self.adaptation_config['adaptation_planner']['planner_config']
            )
            self.adaptation_trigger_criterion = AdaptationTriggerCriterionRegistry(
                criterion_config=self.trigger_criterion_config,
                #For now we assume the criterion config is constant, and any dynamic behviour is internal.
                data_splitting_config=self.data_splitting_config
                #We put this data splitting config here, because it is needed to cross-check that the data splitting
                #has been satisfied when trying to check the adaptation triggering criteria. The data splitting
                #config is an output of the trigger criterion after all.
            )
            self.adaptation_executor = AdaptationExecutor(
                training_tmp_dir=os.path.join(self.experiment_dir, 'adaptation_tmp_dir'),
                completion_dir=os.path.join(self.experiment_dir, 'adaptation_completion_dir'),
                device=self.infer_device,
                max_workers=1
            )
        else:
            raise Exception('Unknown configuration of algorithm state and adaptation flags! Cannot proceed!')
        #####################################################################################################

        self.load() 
        self.build_inference_apps() 
    
    def load(self):
        
        if self.algorithm_state == {}:
            session = PreTrainednnInteractiveSession(
                device=torch.device('cuda', 0),
                use_torch_compile=False,
                verbose=False,
                torch_n_threads=os.cpu_count(),
                do_autozoom=True,
                use_pinned_memory=True
            )
            #Loading from the pretrained model scenario DEFINITELY.  
            app_params = session.initialize_from_trained_model_folder(
                model_training_output_dir=pretrained_ckpt_dir,
                use_fold=0,
                checkpoint_name='checkpoint_final.pth'
            )
        elif self.algorithm_state != {}:
            #In this case, we are definitely in some adaptation scenario, but maybe we don't have an updated algorithm. 
            session = AdaptednnInteractiveSession(
                device=torch.device('cuda', 0),
                use_torch_compile=False,
                verbose=False,
                torch_n_threads=os.cpu_count(),
                do_autozoom=True,
                use_pinned_memory=True
            )
            if self.adaptation_number == 0: 
                #NOTE: Could merge above, but lets be very explicit for clarity. 
                #In this case, we have not adapted yet. 
                # So we will still initialise from the pretrained model. 
                app_params = session.load_from_pretrained_model_folder(
                    model_training_output_dir=pretrained_ckpt_dir,
                    use_fold=0,
                    checkpoint_name='checkpoint_final.pth'
                )   
                #We will create a checkpoint to store this state so that it can be accessed for the 
                #trainer. 
                save_checkpoint(
                    temp_dir=os.path.join(self.experiment_dir, 'storage_temp_kill_dir'),
                    filename='initial_model_configs.pth',
                    desired_dir=self.experiment_dir,
                    configs={
                        'prior_configs':{
                            'input_encoding': app_params['input_encoding'],
                            'model_architecture': app_params['model_architecture'],
                            'input_handling_configs': app_params['input_handling_configs'],
                            'network_configuration': app_params['network_configuration'],
                            'network_weights': session.network.state_dict()
                        }
                    }

                )
                self.checkpoints = {
                    'initial_model': os.path.join(self.experiment_dir, 'initial_model_configs.pth')}
                self.best_ckpt_name = 'initial_model' 
                self.algorithm_training_state.update({
                    'checkpoints': self.checkpoints,
                    'best_ckpt_name': self.best_ckpt_name
                })

            elif self.adaptation_number > 0:
                #In this case, we have adapted before, so we will load from the adapted checkpoint. 
            
                chosen_ckpt = self.checkpoints.get(self.best_ckpt_name) 
                if chosen_ckpt == None or not os.path.exists(chosen_ckpt):
                    raise Exception('The chosen checkpoint for loading the adapted model does not exist! Cannot proceed with loading!')
                app_params = session.load_from_adapted_model_folder(
                    checkpoint_path=chosen_ckpt
                )
                
            else:
                raise Exception('Unknown configuration of algorithm adaptation state! Cannot proceed with loading!') 
        else:
            raise Exception('Unknown configuration of algorithm adaptation state! Cannot proceed with loading!')
        
        self.session = session #We will try to keep the overall API as similar as possible to the pretrained case, 
        #so we can reuse code. 
        #Lets update the app_params with the new info from the loaded session. 
        self.app_params.update(app_params)
    
    def reload_after_adapt(self, checkpoints, best_ckpt_name):
        #Function for reloading the inference session after adaptation has been performed.
        session = AdaptednnInteractiveSession(
                device=torch.device('cuda', 0),
                use_torch_compile=False,
                verbose=False,
                torch_n_threads=os.cpu_count(),
                do_autozoom=True,
                use_pinned_memory=True
            )    
        #In this case, we have adapted before, so we will load from the adapted checkpoint. 
        chosen_ckpt = checkpoints.get(best_ckpt_name) 
        app_params = session.load_from_adapted_model_folder(
            checkpoint_path=chosen_ckpt
        )

        self.session = session #We will try to keep the overall API as similar as possible to the pretrained case, 
        #so we can reuse code. 
        #Lets update the app_params with the new info from the loaded session. 
        self.app_params.update(app_params)


    def extract_registry_config(self, path, name):
        #Function which extracts configs dicts from json or txt files. Takes the path to the file, and the name of the specific config desired.

        if not os.path.exists(path):
            raise Exception(f'The path {path} was not a valid one. Please check.')    

        #Loading the file:
        with open(path) as f:
            configs_registry = json.load(f)
            config = configs_registry[name]

        return config
    
    def app_configs(self):
        #STRONGLY Recommended: A method which returns any configuration specific information for printing to the logfile. Expects a dictionary format.
        return self.app_params 



    def build_inference_apps(self):
        #Building the inference app, needs to have an end to end system in place for each "model" type which can be passed by the request: 
        # 
        # IS_autoseg, IS_interactive_init, IS_interactive_edit. (all are intuitive wrt what they represent.) 
        
        self.infer_apps = {
            'IS_autoseg':{'binary_predict':self.binary_inference},
            'IS_interactive_init': {'binary_predict':self.binary_inference},
            'IS_interactive_edit': {'binary_predict':self.binary_inference}
            }

    def binary_inference(
        self,
        request: dict,
        ) -> torch.Tensor:
        """
        Stub performing **one** forward pass of your model.

        
        bbox : list of dict | None
            Bounding‑box prompt(s).  The dict structure is shown in the challenge
            description; may be absent in refinement iterations.
        clicks : list of dict | None
            Fore‑ and background click dictionaries for every class.
        prev_pred : (D, H, W) np.ndarray | None
            Segmentation from the previous iteration.  May be `None` for the first
            call.

        Returns
        -------

        seg : np.ndarray, dtype=uint8
            Multi‑class segmentation mask.  Background **must** be 0;
            classes start from 1 … N.  Make sure dtype is `np.uint8`.
        """

        init, affine, is_state = self.binary_subject_prep(request)
        self.binary_place_interactions(init, is_state)
        pred, probs_tensor = self.binary_predict() 
        return pred, probs_tensor, affine

    def binary_place_interactions(self, init: bool, is_state: dict):
        #NOTE: nnInteractive performs best when interactions are placed in a sequential order, i.e., each prompt instance is added one after the other.
        # This is because of 2 reasons: 1) the interaction memory is configured to downregulate older interactions, 2) the zoom levels/center are set
        # based on the last interaction placed. Hence, it is expected that the performance will be suboptimal if multiple prompt instances
        # are provided at once. 

        #Given that the foreground points should always be in the vicinity of the target (whereas a background prompt may not be), we will place 
        # the foreground prompt last, so that this is the one that determines the zoom level/center.

        if not bool(is_state):
            raise Exception('Cannot be an interactive request without interaction state! Should not have reached this point!')

        #Extracting the prompt dictionaries from the interaction state.
        p_dict = (is_state['interaction_torch_format']['interactions'], is_state['interaction_torch_format']['interactions_labels'])
        #Determine the prompt types from the input prompt dictionaries
        provided_ptypes = list(set([k for k,v in p_dict[0].items() if v is not None]) & set([k[:-7] for k,v in p_dict[1].items() if v is not None]))
        #Lets provide somewhat of a reasonable limitation, which is that more than one prompt type cannot be provided at once.
        provided_subtypes = set([self.app_params['prompt_subtypes'][ptype] for ptype in provided_ptypes])

        if not len(provided_ptypes) == 1:
            raise Exception('More than one prompt type was provided in the interactive request, cannot proceed with interactive inference!')
        if not len(provided_subtypes) == 1:
            raise Exception('More than 1 prompt subtype was provided, cannot proceed with interactive inference!')
        #Somewhat redundant check, but we will keep it here for now.

        if provided_ptypes[0].title() == "Points":
            points = p_dict[0][provided_ptypes[0]]
            points_lbs = p_dict[1][provided_ptypes[0] + '_labels']

            #Placing the background points first, then the foreground points.
            bg_code = self.configs_labels_dict['background']
            fg_code = self.configs_labels_dict[[k for k in self.configs_labels_dict.keys() if k != 'background'][0]]

            if bg_code != 0:
                raise Exception('Script written assuming background is assigned class 0! Cannot proceed with inference!')

            #First we add the background points, then the foreground. We always want the center of the last interaction to be the foreground class,
            #as this is most likely to be within the region of interest of the target.
            if bg_code in points_lbs:
                bg_idx = (torch.cat(points_lbs) == bg_code).nonzero(as_tuple=True)
                for idx in bg_idx[0]:
                    self.session.add_point_interaction(
                        tuple(points[idx].flatten().tolist()),
                        include_interaction=False,
                        run_prediction=False
                    )
            if fg_code in points_lbs:
                fg_idx = (torch.cat(points_lbs) == fg_code).nonzero(as_tuple=True)
                for idx in fg_idx[0]:
                    self.session.add_point_interaction(
                        tuple(points[idx].flatten().tolist()),
                        include_interaction=True,
                        run_prediction=False
                    )
                
        elif provided_ptypes[0].title() == "Scribbles":
            raise NotImplementedError('Conversion from api-structure to array form not yet implemented')
            
        elif provided_ptypes[0].title() == "Bboxes":   
            #The pre-trained model is trained with 2D bounding boxes in mind. The API will use a convention 
            # that any of the coordinates must be matching. E.g. x_min = x_max, etc, to indicate this. However, the format expected for a 2D bounding box
            # in nnInteractive is to have this represented as the coordinate having difference 1. This is presumably because the bbox is represented as 
            # an array in the network input.

            bboxes = p_dict[0][provided_ptypes[0]]
            bboxes_lbs = p_dict[1][provided_ptypes[0] + '_labels']

            for box in bboxes:
                if not any(box[0, i] == box[0, i+3] for i in range(3)):
                    warnings.warn('nnInteractive natively supports 2D bounding boxes, received a 3D bounding box in the request!')
            #Now we convert the bounding boxes to the expected format, which is to have the bboxes represented by a half-open interval. As opposed to the 
            #closed interval representation used in the API. The upper bound is the open end.
            temp_bboxes = torch.cat(bboxes, dim=0)
            temp_bboxes[:, 3:] += 1 #Converting to half-open interval representation by adding 1 to the upper bounds.
            #Clamping the bounding boxes to be within the image dimensions.
            temp_bboxes[:, :3] = torch.max(torch.zeros(temp_bboxes.shape[0], 3), temp_bboxes[:, :3])
            temp_bboxes[:, 3:] = torch.min((torch.tensor(self.session.original_image_shape[1:]) - 1).unsqueeze(0).repeat(temp_bboxes.shape[0], 1), temp_bboxes[:, 3:])
            #Clamping above from the index! not the shape itself. 

            #Now we will vectorise the process of converting to the expected format.
            #Expected structure of the bboxes for nninteractive input is a list: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            converted_bboxes = torch.stack(
                [temp_bboxes[:,::3], temp_bboxes[:,1::3], temp_bboxes[:,2::3]], dim=1
                ).tolist()  
            #This is very convoluted looking... I know. But it works. The input was structured as [num_bboxes, 6] where the 6 represents:
            # [x_min, y_min, z_min, x_max, y_max, z_max]. We want to convert this to the expected format of nnInteractive. We first stack to reshape
            # the 6 values into 3 pairs. We then convert to list, this creates a nested list of bboxes. 
            
            #Each bbox will have structure [[x_min, x_max], [y_min, y_max], [z_min, z_max]] now, as expected. 

            #Lets handle the bbox labels, first we will make the common sense restriction that each class can only have 1 bounding box per callback. 
            # Not even the most restrictive case (i.e., only one class can have an interaction). 
            
            bin_counts = torch.bincount(torch.stack(bboxes_lbs).flatten())
            if any([count > 1 for count in bin_counts]):
                raise Exception('Each class can only have one bounding box prompt at a given time! Cannot proceed with interactive inference!')
            if bin_counts.shape[0] > 2:
                raise Exception('More than two class labels were provided bounding box prompts OR bbox label was outside of the [0,1] range! Cannot proceed with interactive inference!') 
            
            #First we will look at the background class, then the foreground class, because we want the center of the last interaction to be the foreground class.
            bg_code = self.configs_labels_dict['background']
            
            if bg_code != 0:
                raise Exception('Script written assuming background is assigned class 0! Cannot proceed with inference!')
            if bg_code in bboxes_lbs:
                bg_idx = (torch.cat(bboxes_lbs) == bg_code).nonzero(as_tuple=True)
                if len(bg_idx[0]) > 1:
                    raise Exception('Each class can only have one bounding box prompt at a given time! Cannot proceed with interactive inference!')
                bg_idx = bg_idx[0].item()
                self.session.add_bbox_interaction(
                    converted_bboxes[bg_idx],
                    include_interaction=False,
                    run_prediction=False
                )
            fg_code = self.configs_labels_dict[[k for k in self.configs_labels_dict.keys() if k != 'background'][0]]
            if fg_code in bboxes_lbs:
                fg_idx = (torch.cat(bboxes_lbs) == fg_code).nonzero(as_tuple=True)
                if len(fg_idx[0]) > 1:
                    raise Exception('Each class can only have one bounding box prompt at a given time! Cannot proceed with interactive inference!')
                fg_idx = fg_idx[0].item()
                self.session.add_bbox_interaction(
                    converted_bboxes[fg_idx],
                    include_interaction=True,
                    run_prediction=False
                )
    
        elif provided_ptypes[0].title() == "Lasso":
            raise NotImplementedError('Conversion from api-structure to array form not yet implemented')
        else:
            raise Exception('No other prompting subtypes are supported in nnInteractive.')



    def binary_predict(self):
        '''
        bbox: list[dict] | None,
        clicks: list[dict] | None,
        clicks_order: list[list[str]] | None,
        prev_pred: np.ndarray | None,
        '''
        # now run inference on the last interaction center
        self.session.new_interaction_centers = [self.session.new_interaction_centers[-1]]
        self.session.new_interaction_zoom_out_factors = [self.session.new_interaction_zoom_out_factors[-1]]
        self.session._predict()
        pred = self.session.target_buffer.unsqueeze(0) #Adding back the batch dimension..., we don't assume a one-hot format.
        # del session #We don't delete the session here because we want to keep the application online..
        empty_cache(torch.device('cuda', 0))
        probs_tensor = torch.zeros([2] + list(self.session.target_buffer.shape), dtype=torch.float32) #This is a dummy..they don't give us this. Also its probably going to be deprecated soon, but has not been yet. So just put a dummy.

        return pred, probs_tensor


    def binary_subject_prep(self, request:dict):
        if self.dataset_info is None:
            self.dataset_info = request['dataset_info']
        else:
            if self.dataset_info != request['dataset_info']:
                raise Exception('The dataset info provided in the request does not match the dataset info stored in the algorithm state! Cannot proceed with inference!')
        
        if len(self.dataset_info['task_channels']) != 1:
            raise Exception('The inference app only supports single channel images for segmentation.')
        
        if request['infer_mode'] == 'IS_interactive_edit':
            is_state = request['i_state']
            if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                raise Exception('Cannot be an interactive request without interactive inputs.')
            init = False 

        elif request['infer_mode'] == 'IS_interactive_init':
            is_state = request['i_state']
            if all([i is None for i in is_state['interaction_torch_format']['interactions'].values()]) or all([i is None for i in is_state['interaction_torch_format']['interactions_labels'].values()]):
                raise Exception('Cannot be an interactive request without interactive inputs.')

            init = True
            self.configs_labels_dict = request['config_labels_dict']
            self.load_new_image(request['image']['metatensor'])
            self.session.reset_interactions()
            # self.prev_pred = None  We don't need this. The buffer is already reset.
            empty_cache(torch.device('cuda', 0))
            
        elif request['infer_mode'] == 'IS_autoseg':
            if not self.app_params['autoseg_infer_bool']:
                raise Exception('Autoseg is too OOD for this algorithm (i.e segmentation without any prompts)')
            else:
                NotImplementedError('We have not yet implemented a mechanism for adaptation into autoseg inference!')

        affine = request['image']['meta_dict']['affine']

        return init, affine, is_state 

    def load_new_image(self, image: torch.Tensor):
        self.session.set_image(image.numpy().astype(np.float32))
        target_buffer = torch.zeros(image.shape[1:], dtype=torch.uint8, device='cpu')
        self.session.set_target_buffer(target_buffer)
    

    def clear_untracked_memory_buffer_samples(
        self, 
        memory_buffer_dir: str, 
        tracked_memory_buffer_dict: dict):
        '''
        This function will delete any raw samples and cached samples
        '''
        #We will remove any samples in the memory buffer directory which are not in the tracked memory buffer dictionary.
        all_samples_in_dir = os.listdir(memory_buffer_dir)
        tracked_sample_names = list(tracked_memory_buffer_dict.keys())

        #Find the samples in the dir which are not tracked.
        untracked_sample_names = [i for i in all_samples_in_dir if i not in tracked_sample_names]
        for sample_name in untracked_sample_names:
            #This sample is not tracked, so we will remove it.
            sample_folder = os.path.join(memory_buffer_dir, sample_name)
            shutil.rmtree(sample_folder)

    def clear_untracked_unassigned_samples(
        self, 
        unassigned_samples_list: list, 
        tracked_memory_buffer_dict: dict):
        #We will now remove any samples which are not assigned but also not in the tracked memory buffer dict.
        #This is extremely unlikely to happen on re-run, because any loaded checkpoint should have had these 
        # samples assigned tracked when writing the checkpoint as it is not an io operation. 
        # 
        return [sample_name for sample_name in unassigned_samples_list if sample_name in tracked_memory_buffer_dict.keys()]
       
    def accept_new_sample(self, sample_pair: dict):
        '''
        This is a function which can store a new sample pair which has been annotated (simulates the process of a clinician
        finishing an annotation and the algorithm being able to adapt to this new information/supervision).
        '''
        #Need to pull the final segmentation if a ground truth is not provided (i.e., we are using the final prediction and 
        #assume that the clinician will put no further effort into correcting it).
        if sample_pair['label'] == None:
            raise NotImplementedError('Using the final prediction as the ground truth is not yet implemented.')
        else:
            sample_name = f'buffer_sample_idx_{len(self.samples_memory_buffer_disk)}'
            sample_pair_paths = self.save_sample_on_disk(sample_name, sample_pair)
            #Adding it to the buffer. 
            self.samples_memory_buffer_disk.update({sample_name: sample_pair_paths})
            #We also add it to the unassigned samples, so that when adaptation is triggered these samples can be
            #assigned. 
            self.unassigned_samples.extend([sample_name])

    def save_sample_on_disk(self, sample_name: str, sample_pair: dict):
        memory_buffer_dir = self.meta_algorithm_state.get('memory_buffer_dir', None) 
        if memory_buffer_dir != None and os.path.exists(memory_buffer_dir):
            #Can't save the same sample name twice. Must have cleared any samples saved not in the checkpoint's memory buffer.
            sample_pair_paths = self.data_handler.write_image(sample_pair, memory_buffer_dir, sample_name)
            #Now to save the image and label to the specified paths.
        else:
            raise Exception('Memory buffer directory is not set! Cannot proceed with saving sample to disk!')


        return sample_pair_paths
    
    def trigger_adaptation(self):
        '''
        This is a function which run the adaptation process using the stored samples in the memory buffer.  
        ''' 
        
        #First, lets update the dataset info. For now, we will assume that this CANNOT change, in line with the
        #inference code itself. 
        if self.dataset_info is None:
            raise ValueError('The dataset info in the algorithm state cannot be None when triggering adaptation! Must have been assigned by now. \n' 
                             'Cannot proceed with adaptation!')
        else:
            if self.meta_algorithm_state.get('dataset_info') == None:
                self.meta_algorithm_state['dataset_info'] = self.dataset_info
            elif self.dataset_info != self.meta_algorithm_state.get('dataset_info'):
                raise ValueError('The dataset info in the algorithm state does not match the one in the meta-algorithm state! \n'
                                 'Cannot proceed with adaptation!')
            else:
                pass #They are the same, so no action needed, only put here for clarity. 
            
            if self.meta_algorithm_state.get('configs_labels_dict') == None:
                self.meta_algorithm_state['configs_labels_dict'] = self.configs_labels_dict
            elif self.configs_labels_dict != self.meta_algorithm_state.get('configs_labels_dict'):
                raise ValueError('The config labels dict in the algorithm state does not match the one in the meta-algorithm state! \n'
                                 'Cannot proceed with adaptation!')
            else:
                pass #They are the same, so no action needed, only put here for clarity.

        trigger_condition_met, data_splitting_config = self.adaptation_trigger_criterion(
            meta_algorithm_state=self.meta_algorithm_state
        ) 
        #NOTE: We currently assume that the data splitting config is FIXED. But, we are adding some flexibility here
        #for future changes I guess (honestly this is post-hoc rationalisation from reading spaghetti code). BUT: If we 
        #do have the data splitting config end up being dynamic then we will need to adjust the trigger criterion to 
        #set the data splitting config dynamically also (otherwise it will always revert to the setting it was initialised with
        #which may not be the same as the current one being used!)
        assert data_splitting_config == self.data_splitting_config, 'The data splitting config returned by the adaptation trigger criterion does not match the one stored in the algorithm state! \n'
        'Cannot proceed with adaptation!'
        self.data_splitting_config = data_splitting_config
        
        if trigger_condition_met:
            #NOTE: We will drop the existing inference session here for VRAM considerations. 
            if self.session != None:
                self.session._reset_session()
                self.session = None
                torch.cuda.empty_cache() 

            #Modify the training and validation datalists, and update the unassigned samples.
            #NOTE: Current assumption is that this will only be done when a trigger condition is met.
            #Therefore, the trigger condition also depends on what the data splitting strategies are.
            self.data_split = self.data_handler.data_splitting(
                data_splitting_config=self.data_splitting_config,
                existing_datasplit=self.data_split,
                new_cases_datalist=self.unassigned_samples
            )

            self.unassigned_samples = [] #Clearing the unassigned samples, as they have now been assigned
            #to the data split for adaptation purposes. 

            # We store the data split and also the config because they may have been 
            # updated and we are passing this through for planning. 
            self.meta_algorithm_state.update({
                'data_split': self.data_split,
                'data_splitting_config': self.data_splitting_config,
                'unassigned_samples': self.unassigned_samples
            })

            #Generate plans for training using the meta-algo state, data split and config.
            self.adaptation_plans= self.adaptation_planner.generate_adaptation_plans(
                meta_algorithm_state=self.meta_algorithm_state,
                app_parameters=self.app_params
            )
            self.meta_algorithm_state.update({
                'adaptation_plans': self.adaptation_plans
            })

            if type(self.adaptation_plans) != dict:
                raise TypeError('The generated adaptation plans must be in dictionary format! Cannot proceed with adaptation!')

            #Now let us update the dataloader transforms. 
            self.dataloader_transforms = dict() 
            for split_name, split_dict in self.data_split.items():
                self.dataloader_transforms[split_name] = dict()
                for data_type in split_dict.keys():
                    deterministic_transf = self.adaptation_plans[f'{data_type}_handlers'][split_name]['dataloading_config']['deterministic_transforms']
                    dynamic_transf = self.adaptation_plans[f'{data_type}_handlers'][split_name]['dataloading_config']['dynamic_transforms']
                    assert deterministic_transf is not None and dynamic_transf is not None, 'The dataloader transforms generated in the adaptation plans cannot be None! Cannot proceed with adaptation!'
                    self.dataloader_transforms[split_name][data_type] = {
                        'deterministic_transforms': deterministic_transf,
                        'dynamic_transforms': dynamic_transf
                    }
            self.meta_algorithm_state.update({
                'dataloader_transforms': self.dataloader_transforms
            })
            #Generate dataloaders for adaptation.
            dataloaders = self.data_handler.create_dataloaders(
                data_split=self.data_split,
                memory_buffer_disk=self.samples_memory_buffer_disk, #The dict containing the filepaths for the samples stored on disk.
                adaptation_planner=self.adaptation_plans,
                memory_buffer_dir=self.samples_memory_buffer_dir
            )
            if type(dataloaders) != dict:
                raise TypeError('The generated dataloaders must be in dictionary format! Cannot proceed with adaptation!')
              
            #Then run adaptation procedure here.
            training_state = self.adaptation_executor(
                dataloaders=dataloaders,
                adaptation_plans=self.adaptation_plans,
                checkpoints=self.checkpoints,
                adaptation_number=self.adaptation_number,
                configs_labels_dict=self.configs_labels_dict
            )
            #Updating adaptation dependent info.
            self.checkpoints.update(training_state['checkpoints'])
            self.best_ckpt_name = training_state['best_ckpt_name'] 
            self.adaptation_number += 1
            self.meta_algorithm_state.update({
                'adaptation_number': self.adaptation_number}
            )
            self.algorithm_training_state.update({
                'checkpoints': self.checkpoints,
                'best_ckpt_name': self.best_ckpt_name
            })

            #Updating the session and app params after adaptation.
            self.update_session_and_app_params()
            
            #We write the state after adaptation, just for sanity checking and seeing how the algorithm 
            #being used has evolved. 
            self.write_state = True
            self.meta_algorithm_state.update({
                'write_state': self.write_state
            })

        else:
            self.write_state = False  
            self.meta_algorithm_state.update({
                'write_state': self.write_state
            })
        return self.update_algorithm_state_info()

    def update_session_and_app_params(self):
        #Function which updates the app params after adaptation and resets the session accordingly. 
        
        #First we will reset and remove the existing session if this hasn't already been done.  
        if self.session is not None:
            self.session._reset_session()
            self.session = None
            torch.cuda.empty_cache() 

        if self.adaptation_plans['algorithm_config']['functionality_adaptation'] == None:
            #We will not update any app params for now. We are just going to take an existing app and optimise it. 
            self.app_params = self.app_params

            session = AdaptednnInteractiveSession(
                device=torch.device('cuda', 0),
                use_torch_compile=False,
                verbose=False,
                torch_n_threads=os.cpu_count(),
                do_autozoom=True,
                use_pinned_memory=True
            )
            #Lets now load the checkpoint into the session and initialise a new one.
            checkpoint_path = self.checkpoints.get(self.best_ckpt_name) 
            if checkpoint_path == None:
                raise Exception('The best checkpoint name does not correspond to any stored checkpoint path! Cannot proceed with reloading session after adaptation!')
            if not os.path.exists(checkpoint_path):
                raise Exception('The checkpoint path for the best checkpoint does not exist on disk! Cannot proceed with reloading session after adaptation!')
            
            session.load_from_adapted_model_folder(
                checkpoint_path=self.checkpoints.get(self.best_ckpt_name)
            )
            self.session = session 
            
        else:
            raise NotImplementedError('Updating app params after functionality adaptation is not yet implemented!') 
                
        
    def update_algorithm_state_info(self):
        '''
        This is a function which returns any information regarding the state of the algorithm, which may be necessary for
        continuation of execution later on down the line. E.g., the checkpoint location containing:
        model-weights, optimizer state, learning rate scheduler state,
        '''

        #We are being borderline paranoid here, but we want to make sure that we are not missing any untracked info.
        meta_algorithm_state = {
            'algo_cache_name': self.algo_cache_name,
            'adaptation_config': self.adaptation_config,
            'dataset_info': self.dataset_info,
            'experiment_dir': self.experiment_dir,
            'memory_buffer_dir': self.samples_memory_buffer_dir,
            'memory_buffer_disk': self.samples_memory_buffer_disk,
            'trigger_criterion_config': self.trigger_criterion_config,
            'data_split': self.data_split,
            'dataloader_transforms': self.dataloader_transforms,
            'data_splitting_config': self.data_splitting_config,
            'unassigned_samples': self.unassigned_samples,
            'adaptation_plans': self.adaptation_plans,
            'app_params': self.app_params,
            'write_state': self.write_state,
            'adaptation_number': self.adaptation_number,
            'configs_labels_dict': self.configs_labels_dict
            }
        assert all([meta_algorithm_state[i] == self.meta_algorithm_state[i] for i in meta_algo_state_keys]), 'The meta algorithm state being updated does not match the stored meta algorithm state! Cannot proceed with updating algorithm state info!'
        algorithm_training_state = {
            'checkpoints': self.checkpoints,
            'best_ckpt_name': self.best_ckpt_name,
            }
        assert all([algorithm_training_state[i] == self.algorithm_training_state[i] for i in training_state_keys]), 'The algorithm training state being updated does not match the stored algorithm training state! Cannot proceed with updating algorithm state info!'

        if [i for i in self.meta_algorithm_state.keys() if i not in meta_algo_state_keys]:
            raise Exception('The keys in the meta algorithm state being updated do not match the expected keys! Cannot proceed with updating algorithm state info!')
        if [i for i in self.algorithm_training_state.keys() if i not in training_state_keys]:
            raise Exception('The keys in the algorithm training state being updated do not match the expected keys! Cannot proceed with updating algorithm state info!')
        return {
            'meta_algorithm_state': meta_algorithm_state,
            'algorithm_training_state': algorithm_training_state
        }
    
    def __call__(self, request:dict):

        if len(request['config_labels_dict']) == 2:
            class_type = 'binary'
        elif len(request['config_labels_dict']) > 2:
            class_type = 'multi'
            raise NotImplementedError('See the SegFM implementation for integrating multi-class segmentation interpretation.')
        else:
            raise Exception('Should not have received less than two class labels at minimum')
        
        #We create a duplicate so we can transform the data from metatensor format to the torch tensor format compatible with the inference script.
        modif_request = copy.deepcopy(request) 

        app = self.infer_apps[modif_request['infer_mode']][f'{class_type}_predict']

        #Setting the configs label dictionary for this inference request.
        self.configs_labels_dict = modif_request['config_labels_dict']


        pred, probs_tensor, affine = app(request=modif_request)

        pred = pred.to(device='cpu')
        probs_tensor = probs_tensor.to(device='cpu')
        # affine = affine.to(device='cpu')
        torch.cuda.empty_cache()

        assert probs_tensor.shape[1:] == request['image']['metatensor'].shape[1:]
        assert pred.shape[1:] == request['image']['metatensor'].shape[1:] 
        assert torch.all(affine == request['image']['meta_dict']['affine'])
        assert isinstance(probs_tensor, torch.Tensor) 
        assert isinstance(pred, torch.Tensor)
        assert isinstance(affine, torch.Tensor)

        output = {
            'probs':{
                'metatensor':probs_tensor,
                'meta_dict':{'affine': affine}
            },
            'pred':{
                'metatensor':pred,
                'meta_dict':{'affine': affine}
            },
        }
        #Functionally probably wont do anything but putting it here as a placebo. Won't make a diff because there are references
        #to these variables throughout.
        del pred 
        del probs_tensor
        del affine
        del modif_request
        # torch.cuda.empty_cache() 
        empty_cache(torch.device('cuda', 0))

        return output


# if __name__ == '__main__':
   
#     infer_app = InferApp(
#         infer_device=torch.device('cuda', index=0)
#         )

#     infer_app.app_configs()

#     from monai.transforms import LoadImaged, Orientationd, EnsureChannelFirstd, Compose 
#     import nibabel as nib 

#     input_dict = {
#         'image' :os.path.join(app_local_path, 'debug_image/BraTS2021_00266.nii.gz')
#         }    
#     load_and_transf = Compose([LoadImaged(keys=['image'], image_only=True), EnsureChannelFirstd(keys=['image']), Orientationd(keys=['image'], axcodes='RAS')])

#     loaded_im = load_and_transf(input_dict)
#     input_metatensor = torch.from_numpy(loaded_im['image'].array)
#     meta = {
#         'original_affine': copy.deepcopy(torch.from_numpy(loaded_im['image'].meta['original_affine']).to(dtype=torch.float64)), 
#         'affine': copy.deepcopy(loaded_im['image'].meta['affine']).to(dtype=torch.float64)}
    
#     request = {
#         'image':{
#             'metatensor': input_metatensor,
#             'meta_dict':meta
#         },
#         # 'infer_mode':'IS_interactive_edit',
#         'infer_mode': 'IS_interactive_init',
#         'config_labels_dict':{'background':0, 'tumor':1},
#         'dataset_info':{
#             'dataset_name':'BraTS2021_t2',
#             'dataset_image_channels': {            
#                 "T2w": "0"
#             },
#             'task_channels': ["T2w"]
#         },
#         'i_state':
#             {
#             'interaction_torch_format': {
#                 'interactions': {
#                     'points': None, #[torch.tensor([[40, 103, 43]]), torch.tensor([[61, 62, 39]])], #None 
#                     'scribbles': None, 
#                     'bboxes': [torch.Tensor([[56,30,17, 92, 76, 51]]).to(dtype=torch.int64)] #None 
#                     },
#                 'interactions_labels': {
#                     'points_labels': None, # [torch.tensor([0]), torch.tensor([1])], #None,
#                     'scribbles_labels': None, 
#                     'bboxes_labels': [torch.Tensor([1]).to(dtype=torch.int64)] #None
#                     }
#                 },
#             'interaction_dict_format': {
#                 'points': None, 
#                 # {
#                     # 'background': [[40, 103, 43]],
#                     # 'tumor': [[61,62,39]]
#                     #},  
#                 'scribbles': None,
#                 'bboxes': {'background': [], 'tumor': [[56,30,17, 92, 76, 51]]} #None
#                 },    
#         },
#     }
#     output = infer_app(request)
#     print('halt')
