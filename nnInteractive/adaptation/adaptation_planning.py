'''
This is a script which contains functions for configuring the adaptation algorithm. It configures the
training pipeline, typically according to the current state of the meta-algorithm.

Can be also configured based on criteria such as performance metrics, dataset/task characteristics/task-specific
characteristics etc.

'''
import os
import sys
import numpy as np
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()
# app_local_path = os.path.abspath(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))
from nnInteractive.adaptation.adaptation_trigger_criteria import AdaptationTriggerCriterionRegistry

class AdaptationPlanner:
    def __init__(
        self,
        planner_name: str,
        planner_config: dict,
        *args,
        **kwargs):
        '''
        This is a class which generates adaptation plans/configures the training pipeline based on the 
        current state of the algorithm/meta-algorithm and training history.

        inputs:
        - planner_config: dict
            This is a dictionary which contains some information about configuring the auto-adaptive planner.
            E.g., which strategies to use in the training pipeline, or what algorithms to use in choosing
            adaptation strategies for use in the training pipeline etc. it doesn't have to be only a single
            configuration, it could be a heuristic or set of rules to use when generating the planner.

            nnU-Net/autoML inspired design. 
        
        outputs:
        - adaptation_plan: dict
            This is a dictionary which contains the adaptation plan/configuration for the training pipeline
            class. 
                Recommended fields: Dataloading configs, interaction simulation strategy, compatible 
                functionality adaptation strategy, model architecture, loss function, optimizer, learning rate, 
                scheduling, etc. Corresponding hyperparameters.  
        '''
        if planner_name not in planner_registry:
            raise ValueError(f'Invalid adaptation planner: {planner_name}. Permitted planners are: {list(planner_registry.keys())}')
        self.planner_generator = planner_registry[planner_name](
            planner_config
        )

    def generate_adaptation_plans(
        self,
        meta_algorithm_state: dict, 
        app_parameters: dict,
        *args,
        **kwargs) -> dict:
        '''
        inputs:
            - meta_algorithm_state: dict
                This is a dictionary which contains the current state of the meta-algorithm.
            - app_parameters: dict
                This is a dictionary which contains the application parameters, slightly different from 
                the meta-algorithm state. It contains additional information regarding the functionalities of
                the application etc. 
        '''
        return self.planner_generator(
            meta_algorithm_state,
            app_parameters,
            *args,
            **kwargs
        )

class Prototype_Static_Planner:
    def __init__(self,
        planner_config: dict):

        self.planner_config = planner_config
        
        self.adaptation_plan = {
        }
            
        
        self.adaptation_utils = {
            'epoch_config': self.determine_epoch_config,
            'algorithm_config': self.determine_algorithm_config,
            'train_handlers': self.determine_train_handler_config,
            'val_handlers': self.determine_val_handler_config
        }

    def determine_epoch_config(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        *args,
        **kwargs
        ) -> int:
        '''
        This is a function which determines the configuration for epochs in the training pipeline.
        E.g., max_epochs, etc.
        '''
        epoch_config = {
            'max_epochs': 10 #hardcoding it for now.
        }
        return epoch_config
    
    def determine_algorithm_config(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        *args,
        **kwargs
        ) -> dict:
        '''
        This is a function which determines the algorithm configuration for the training pipeline.
        '''
        #This pertains to the actual functionalities of the algorithm, which is more general than
        #just training/validation.

        #NOTE: For now, we are just using the same configuration but trying to make it more efficient
        algo_conf = {
            'input_encoding': 'nnInteractiveUNetEncoding',
            'input_handling_configs': app_parameters.get('input_handling_configs'),
            'functionality_adaptation': None, #We are just adapting an interactive method to be more efficient
            #for now.
            'model_architecture': 'nnInteractiveUNetFrozen', #'nnInteractiveUNet',
            'network_configuration': app_parameters.get('network_configuration'),
            }
        return algo_conf
     

    def determine_train_handler_config(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        *args,
        **kwargs
        ) -> dict:
        '''
        This is a function which determines the training handler configuration for the training pipeline.
        It can call on the sub-functions to determine the sub-configurations.
        '''
        config_callbacks = {
        'dataloading_config': self.determine_training_dataloading_config,
        'prompter': self.determine_training_prompter,
        'loss_config': self.determine_loss_config,
        'optimisation_config': self.determine_optimisation_config
            }
        
        train_handlers = dict()
        for split_name, split_dict in meta_algorithm_state['data_split'].items():
            train_handlers[split_name] = dict()
            train_handlers[split_name] = {k: v(meta_algorithm_state, app_parameters, split_dict['train']) for k, v in config_callbacks.items()}

        return train_handlers
    
    def determine_val_handler_config(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        *args,
        **kwargs
        ) -> dict:
        '''
        This is a function which determines the validation handler configuration for the training pipeline.
        It can call on the sub-functions to determine the sub-configurations.
        '''
        config_callbacks =  {
            'dataloading_config': self.determine_validation_dataloading_config,
            'prompter': self.determine_validation_prompter,
            'performance_tracking_config': self.determine_performance_tracking_config
        }

        val_handlers = dict()
        for split_name, split_dict in meta_algorithm_state['data_split'].items():
            val_handlers[split_name] = {k: v(meta_algorithm_state, app_parameters, split_dict['val']) for k, v in config_callbacks.items()}

        return val_handlers 
    
    def determine_training_dataloading_config(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        split: list[str],
        ) -> dict:
        '''
        This is a function which determines the dataloading configuration for the training pipeline.
        '''
        #We set some parameters for determining the number of samples per batch item based on VRAM constraints a
        #in addition to the number of iterations per epoch to get a sufficient number of variation seen per epoch.
        patch_limit = 2 #Setting a hard limit on the number of patches per batch callback based on VRAM constraints.
        batch_size = 2 #Temporarily hardcoded.
        total_iterations = 50 #Lets hard code this for now, we can make it dynamic based on dataset size later.
        #We are going to treat the iterations here as a process of sampling from a patch generation distribution.
        num_samples_per_batch_item = max(1, patch_limit // batch_size)
        
        #NOTE: Temporarily static while building mvp
        updated_dataloading_configs = {
            'batch_size': batch_size, #Min 2, max whatever we can fit on the GPU. We should probably add this
            #TODO: Temporarily hardcoded. Needs to be adjusted probably depending on
            #available VRAM/availability of samples.  E.g., using mini-batch to introduce stochasticity/regularising
            #effects for small sample sizes.
            'total_iterations': total_iterations, 
            'num_workers': 1,
            'pin_memory': True,
            'deterministic_transforms':{
                'LoadImaged': {
                    'keys': ['image', 'label'],
                    'reader': 'ITKReader',
                    'dtype': np.float32,
                    'meta_keys': None,
                    'meta_key_postfix': DEFAULT_POST_FIX,
                    'overwriting': False,
                    'image_only':True,
                    'ensure_channel_first': False,
                    'simple_keys':False,
                    'prune_meta_pattern': None,
                    'prune_meta_sep': ".",
                    'allow_missing_keys': False,
                    'expanduser': True,

                    },
                'EnsureChannelFirstd': {
                    'keys': ['image', 'label'],
                    'strict_check': False,
                    'allow_missing_keys': False,
                    'channel_dim': None
                    }
            },
            'dynamic_transforms':{
                # TODO: Uncomment and add the rest of the transforms. 
                'RandCropByPosNegLabeld': {
                    'keys': ['image', 'label'],
                    'label_key': 'label',
                    'spatial_size': app_parameters.get('patch_size', (192, 192, 192)),
                    'pos': 1,
                    'neg': 1,
                    'num_samples': num_samples_per_batch_item,
                    'image_key': None,
                    'image_threshold': 0.0,
                    'allow_smaller': True,
                    'allow_missing_keys': False,
                    'lazy': False,
                    'fg_indices_key': None,
                    'bg_indices_key': None,
                    },
                'DivisiblePadd': {
                    'keys': ['image', 'label'],
                    'k': 192,
                    'mode': 'constant',
                    'method': 'symmetric',
                    'allow_missing_keys': False,
                    'lazy': False,
                    }
                }
        }
        return updated_dataloading_configs
    
    def determine_validation_dataloading_config(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        split: list[str],
        ) -> dict:
        '''
        This is a function which determines the dataloading configuration for the validation pipeline.
        '''
        #We will set some of the parameters for determining the number of "iterations" based off a necessity for
        #statistical power in validation metrics.
        patch_limit = 5 #Setting a hard limit on the number of patches per batch callback based on VRAM constraints.
        num_patches = 30 #Lets set 30 patches for validation for now.
        batch_size = 1 #Temporarily hardcoded.
        num_samples_per_batch_item = max(1, patch_limit // batch_size) #We will need to ensure that on small dataset sizes that we have sufficient
        #statistical power to calculate validation metrics properly.
        total_iterations = max(1, num_patches // (batch_size * num_samples_per_batch_item)) #Total iterations here will not mean updates, but rather reflect the total 
        
        
        updated_dataloading_configs = {            
            'batch_size': batch_size, #Temporarily hardcoded. Needs to be adjusted probably depending on
            #available VRAM/availability of samples. And patch-subsampling if used will also have an effect. We want
            #a reasonable measure of performance, which can suffer if samples are correlated (subsampling) or
            #batch size is too small.
            'num_workers': 1,
            'total_iterations': total_iterations,
            'pin_memory': True,
            'metrics': True,
            'deterministic_transforms':{
                'LoadImaged': {
                    'keys': ['image', 'label'],
                    'reader': 'ITKReader',
                    'dtype': np.float32,
                    'meta_keys': None,
                    'meta_key_postfix': DEFAULT_POST_FIX,
                    'overwriting': False,
                    'image_only':True,
                    'ensure_channel_first': False,
                    'simple_keys':False,
                    'prune_meta_pattern': None,
                    'prune_meta_sep': ".",
                    'allow_missing_keys': False,
                    'expanduser': True,

                    },
                'EnsureChannelFirstd': {
                    'keys': ['image', 'label'],
                    'strict_check': False,
                    'allow_missing_keys': False,
                    'channel_dim': None
                    },
                
                },
            'dynamic_transforms':{
                #We probably won't perform on-the-fly validation constantly using full resolution/FOV as it 
                # would take too long. We will probably perform it on patches, so clearly there will be a divergence
                #between validation performance here and actual performance on the full res/FOV.

                'RandCropByPosNegLabeld': {
                    'keys': ['image', 'label'],
                    'label_key': 'label',
                    'spatial_size': app_parameters.get('patch_size', (192, 192, 192)),
                    'pos': 1,
                    'neg': 1,
                    'num_samples': num_samples_per_batch_item,
                    'image_key': None,
                    'image_threshold': 0.0,
                    'allow_smaller': True,
                    'allow_missing_keys': False,
                    'fg_indices_key': None,
                    'bg_indices_key': None,
                    'lazy': False,
                    }, 
                'DivisiblePadd': {
                    'keys': ['image', 'label'],
                    'k': 192,
                    'mode': 'constant',
                    'method': 'symmetric',
                    'allow_missing_keys': False,
                    'lazy': False,
                    }
                }, 
            }
        return updated_dataloading_configs 
    
    def determine_training_prompter(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        split: list[str],
        ) -> dict:
        '''
        This is a function which determines the inner loop mechanism configuration 
        for training.
        '''
        prototype_prompter = \
            {
            "methods":{
                "points":["uniform_random"],
                "scribbles":None,
                "bboxes":None,
                "lassos":None				
                },
            "build_params":{
                "points":{
                    "uniform_random": {
                        "n_max": 1
                        }
                    },
                "scribbles":None,
                "bboxes":None,
                "lassos":None				
                },
            "mixture_params":None,
            "prompter_class_type":"RandomAgent"
            }
        
        prompt_conf = {
                'mode_configs': {
                    'Interactive Init': {
                        'prompter': prototype_prompter 
                    },
                    'Interactive Edit': {
                        'prompter': prototype_prompter
                    } 
                },
                'use_mem': False, #Whether to use memory of past interactions in the inner loop to condition the prompter. 
                'num_loop': 10, #Number of edit iters/inner loops in the inner loop
                } 
        return prompt_conf  #Currently static, no special treatment for now.7

    def determine_validation_prompter(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        split: list[str],
        ) -> dict:
        '''
        This is a function which determines the inner loop mechanism configuration 
        for validation.
        '''
        prototype_prompter = \
            {
            "methods":{
                "points":["uniform_random"],
                "scribbles":None,
                "bboxes":None,
                "lassos":None				
                },
            "build_params":{
                "points":{
                    "uniform_random": {
                        "n_max": 1
                        }
                    },
                "scribbles":None,
                "bboxes":None,
                "lassos":None				
                },
            "mixture_params":None,
            "prompter_class_type":"RandomAgent"
            }
        
        prompt_conf = {
                'mode_configs': {
                    'Interactive Init': {
                        'prompter': prototype_prompter 
                    },
                    'Interactive Edit': {
                        'prompter': prototype_prompter
                    } 
                },
                'use_mem': False, #Whether to use memory of past interactions in the inner loop to condition the prompter. 
                'num_loop': 10, #Number of edit iters/inner loops in the inner loop
                } 
        return prompt_conf  #Currently static, no special treatment for now.7

    def determine_loss_config(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        split: list[str],
        ) -> dict:
        '''
        This is a function which determines the loss function configuration for the training pipeline.
        '''
        num_output_channels = self.adaptation_plan['algorithm_config']['network_configuration'].get('num_output_channels')
        if num_output_channels is None:
            raise RuntimeError('num_output_channels must be specified in the network_configuration to determine the loss configuration!')
        if num_output_channels == 1:
            sigmoid=True
            softmax=False
        elif num_output_channels > 1:
            sigmoid=False
            softmax=True
        else:
            raise RuntimeError('num_output_channels must be at least 1 to determine the loss configuration!')
        loss_conf = {
            'base': { #This is the base loss function used to calculate a loss at each interaction event.
                'name': 'DiceCELoss', 
                'params': {
                    'Dice':{
                        'include_background':False, 
                        'to_onehot_y':True, 
                        'sigmoid':sigmoid, 
                        'softmax':softmax,
                        'other_act':None, 
                        'squared_pred':False, 
                        'jaccard':False, 
                        'reduction':'mean', 
                        'smooth_nr':1e-05, 
                        'smooth_dr':1e-05, 
                        'batch':False, 
                        'weight':None
                    }, #The cross entropy losses take logits as input, so no need to specify anything special here.
                    'CrossEntropy':{
                        'weight': None, 
                        'size_average': True, 
                        'ignore_index': -100, 
                        'reduce': True, 
                        'reduction': 'mean', 
                        'label_smoothing': 0.0
                    },
                    'BinaryCrossEntropy':{
                        'weight': None,
                        'size_average': True,
                        'reduce': True,
                        'reduction': 'mean',
                        'pos_weight': None,
                    },
                'weight':{'CE': 1.0, 'BCE': 1.0, 'Dice': 1.0} #Equal weighting for both loss components.
                },
                
            },
            'wrapper_config': { #This is a wrapper around the base loss to handle the refinement process. 
                'used_outputs': 'all',
                'merge_strategy': 'mean', #Could be 'mean', 'sum', 'weighted_sum' etc.
                'early_exit_padding_strategy': 'Unpadded' #Could be None (i,e, don't pad and just use the actual ones),
                # 'last' (i.e. pad using the last loss value), just intended for batch size > 1 where different 
                #samples may have different number of interactions. 
                } 
            } #Currently static, no special treatment for now.

        return loss_conf 

    def determine_optimisation_config(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        split: list[str],
        ) -> dict:
        '''
        This is a function which determines the optimisation configuration for the training pipeline.
        '''
        optimisation_config = {
            'optimiser': self.determine_optimiser(meta_algorithm_state=meta_algorithm_state, app_parameters=app_parameters, split=split),
            'learning_rate_config': self.determine_learning_rate_config(meta_algorithm_state=meta_algorithm_state, app_parameters=app_parameters, split=split)
            }  
        return optimisation_config #Currently static because it calls on static methods, no special treatment for now.
    
    def determine_optimiser(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        split: list[str],
        ) -> dict:
        '''
        This is a function which determines the optimiser for the training pipeline.
        '''
        optimiser_conf = {
            'name': 'adam',
            'params': {
                'lr': 0.001,
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'weight_decay': 1e-5,
                'amsgrad': False,
                'foreach': None,
                'maximize': False,
                'capturable': False,
                'differentiable': False,
                'fused': None
                }
        }
        
        return optimiser_conf 
                
    def determine_learning_rate_config(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        split: list[str],
        ) -> dict:
        '''
        This is a function which determines the learning rate configurations for the training pipeline.
        '''
        #TODO:
        #Potentially put some logic here to adjust learning rate depending on the state of the model, available
        # data samples etc.       
        lr_conf = {
            'scheduler_configs': {
                'stepLR':{                  
                'scheduler_params':{
                    'step_size': 10,
                    'gamma': 0.1,
                    'verbose': False,
                    'last_epoch': -1,
                    },
                }
            },
            'scheduler_order_config': [{'name': 'stepLR'}] #Just information about the order of schedulers to use. 
        } 
        return lr_conf 
    
    def determine_performance_tracking_config(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        split: list[str],
        ) -> dict:
        '''
        This is a function which determines the performance tracking configuration for the validation pipeline.
        '''
        
        metrics_conf = {
            'per_iter_train_metric_config': {
                'DiceMetric': {
                    'params': {
                        'include_background': False,
                        'classwise_agg': True,
                        'batchwise_agg': False,
                        'ignore_empty': False
                        },  
                    'wrapper': {
                        'iteration': 'final',
                        'write': False #We do not want to write this to tensorboard
                    },
                },
            },
            'per_iter_val_metric_config': {
                'DiceMetric': {
                    'params': {
                        'include_background': False,
                        'classwise_agg': True,
                        'batchwise_agg': False,
                        'ignore_empty': False
                        },
                    'wrapper': {
                        'iteration': 'all',
                        'write': False #We do not want to write this to tensorboard
                    },
                },
            },
            'aggregate_train_metric_config': {
                # 'DiceAUCMetric': {
                #     'params': {
                #         'batchwise_reduce': False,
                #         'iteration': 'all'
                #         },
                #     'wrapper': {
                #         'base_metric': 'DiceMetric',
                #         'iteration': 'all', #We need all iterations possible for train aggregation.
                #         'write': False #We do not want to write this to tensorboard
                #     },
                # },
            },
            'aggregate_val_metric_config': {
                'DiceAUCMetric': {
                    'params': {
                        'batchwise_reduce': False,
                        'iteration': 'all'
                        },
                    'wrapper': {
                        'base_metric': 'DiceMetric',
                        'iteration': 'all', #We need all iterations possible for val aggregation.
                        'write': True #We want to write this to tensorboard
                    },
                },
            },
            'improvement_criterion_config': {
                'metric_name': 'DiceAUCMetric',
                'metric_type': 'aggregate',  #Could be 'per_iter' or 'aggregate'
                'ema': True,
                'criterion': 'strictly_greater'  #Could be 'strictly_greater', 'greater_equal'
                }
            } 
        
        return metrics_conf
    def __call__(
        self,
        meta_algorithm_state: dict,
        app_parameters: dict,
        *args,
        **kwargs
        ) -> dict:
        '''
        This is a simple adaptation planner which returns a static adaptation plan/configuration for the training pipeline.
        '''
        for k, v in self.adaptation_utils.items():
            self.adaptation_plan.update(
                {k: v(meta_algorithm_state, app_parameters, *args, **kwargs)}
            )
        return self.adaptation_plan
    




planner_registry = {
    'Prototype_Static_Planner': Prototype_Static_Planner,
}