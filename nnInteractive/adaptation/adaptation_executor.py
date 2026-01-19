import os
import sys
# app_local_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# sys.path.append(app_local_path)
from nnInteractive.adaptation.training_utils.trainer import Trainer
from nnInteractive.adaptation.training_utils.saving_utils import save_checkpoint
import torch
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import shutil
from torch.utils.tensorboard import SummaryWriter

def _setup_logging(level: int = logging.INFO) -> logging.Logger:
    # logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(__name__)

class AdaptationExecutor:
    """Callable executor for executing adaptation plans.

    Usage patterns:
        - Provide plans directly: executor(plans)
        - Configure a validator function which checks that all of the plans configs are supported.
        - Configure a callback which puts together the plans configs into the structure required for 
        executing training.
        - return outcomes of training. 
        - should be capable of auto-detecting incomplete job executions. 
        
    The callback can be any callable that accepts zero args or a single `config`
    dict argument and returns a dict mapping plan_id -> plan_dict.
    """

    def __init__(
        self,
        training_tmp_dir: str,
        completion_dir: str, 
        device: str = "cuda:0", 
        max_workers: int = 1):
        """
        inputs:
        training_tmp_dir: str: directory relative to cache dir to save/load training checkpoints and
        other files during the training process. It is only tempoarily used during adaptation.
        device: str: device to run adaptation on (e.g., "cpu", "cuda:0").
        max_workers: int: maximum parallel workers to use for adaptation. 
        
        """
        self.training_tmp_dir = training_tmp_dir
        self.training_completion_dir = completion_dir
        self.device = device
        self.max_workers = max_workers
        self.logger = _setup_logging()

        self.required_epoch_configs = [
            'max_epochs'
        ]
        self.required_algorithm_configs = [
            'functionality_adaptation',
            'model_architecture',
        ]
        self.required_train_configs = [
            'prompter',
            'loss_config',
            'optimisation_config'
        ]
        self.required_val_configs = [
            'prompter', 
            'performance_tracking_config'
        ]

    def configure_callback(self, plan_config: dict):
        """Configure a callback used to execute plans.
        The callback must be callable and must accept a single argument: the 
        `adaptation_plans` dict.

        This generates an error if the callback is not callable.
        """

        #Now we form the callable based on plan_config. 
        cb = Trainer(
            planner_config= {
                'epoch_config': plan_config.get('epoch_config'),
                'algorithm_config': plan_config.get('algorithm_config'),
                'train_handlers': plan_config.get('train_handlers'),
                'val_handlers': plan_config.get('val_handlers'),
            },
            device=self.device)

        if not callable(cb):
            raise TypeError("callback must be callable")
        self._callback = cb

    def _validate_configs_for_callback(self, planner: Dict[str, Any]) -> None:
        """Ensure required components (if any) are present in plans."""
        
        #First we validate that the algorithm configs are present and supported.
        missing = [k for k in self.required_algorithm_configs if k not in planner['algorithm_config']]
        if missing:
            raise KeyError(f"Missing required plan components: {missing}")
        #Next we validate that the epoch configs are present and supported.
        missing = [k for k in self.required_epoch_configs if k not in planner['epoch_config']]
        if missing:
            raise KeyError(f"Missing required plan components: {missing}")
        #Next we validate that the train and val handlers are present and supported.
        for split, config in planner['train_handlers'].items():
            missing = [k for k in self.required_train_configs if k not in config]
            if missing:
                raise KeyError(f"Missing required plan components: {missing} in split {split}")
        for split, config in planner['val_handlers'].items():
            missing = [k for k in self.required_val_configs if k not in config]
            if missing:
                raise KeyError(f"Missing required plan components: {missing} in split {split}")
        

    def __call__(
        self, 
        dataloaders: dict, 
        adaptation_plans: Dict[str, Dict[str, Any]],
        checkpoints: Dict[str, str],
        adaptation_number: int,
        configs_labels_dict: Dict[str, int] 
        ) -> dict:
        """Invoke the configured callback with the provided plans.
        The same plans are used across all dataloaders provided (splits), that is the current assumption. 
        The executor validates the plans and that all required callback
        components are present before calling the callback as: `callback(plans)`.

        The callback is responsible for executing training/adaptation and returns a dict containing training
        state information.

        inputs:
        dataloaders: dict: dictionary mapping split names to dataloader dicts.
        adaptation_plans: Dict[str, Dict[str, Any]]: dictionary configuring the adaptation plans. assumed
        to be fixed across all splits.
        checkpoints: Dict[str, str]: dictionary mapping split names to checkpoint paths to resume from.
        adaptation_number: int: integer indicating the current adaptation iteration number.
        configs_labels_dict: Dict[str, int]: dictionary mapping class configuration labels to integers.
        """
        if adaptation_plans is None or adaptation_plans == {}:
            raise RuntimeError("No adaptation plans provided to executor.")
        # ensure required components for the callback exist
        self._validate_configs_for_callback(adaptation_plans) 
        #check whether existing incomplete executions are present and handle them
        
        #Configuring the trainer. We currently assume the same plans are used across all splits! This
        #may be subject to change later.
        self.configure_callback(adaptation_plans) 
        
        if list(checkpoints.keys()) == ['initial_model']:
            #In this case, we have only the initial model provided. 
            #We will assume that all splits use the same initial model.
            initial_model_ckpt = checkpoints.get('initial_model')
            checkpoints.update({split_name: initial_model_ckpt for split_name in dataloaders.keys()})
        

        for split_name, dataloader_dict in dataloaders.items():
            self.logger.info('Starting adaptation for split: {}'.format(split_name))

            if not os.path.exists(
                os.path.join(
                    self.training_completion_dir, 
                    f'adaptation_{adaptation_number}', 
                    f"completed_{split_name}", 'best.pth'
                    )
                ): #Checking that the checkpoint has safely made it.
            
                #In this case, we haven't managed to finish the training for this split yet. 

                #Lets inspect the training state pkl file. 
                if os.path.exists(
                    os.path.join(
                        self.training_tmp_dir, 
                        f'adaptation_{adaptation_number}', 
                        split_name, 
                        'last.pth'
                        )
                    ): 
                    #We may be resuming training assuming we didn't exit early before even starting.
                    with open(os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name, 'last.pth'), 'rb') as f:
                        stored_state = torch.load(f, weights_only=False)
                        if stored_state['plans'] != adaptation_plans:
                            raise RuntimeError(f"Stored plans in {os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name, 'last.pth')} do not match the provided plans! \n"
                            "Cannot resume training during adaptation.")
                        
                        if not os.path.exists(os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name, 'best.pth')): 
                            #Somehow the last.pth initialised, but not the best.pth. this should typically only be possible if current epoch is still 0, and 
                            #we never even managed to finish the zeroeth validation epoch.
                            if stored_state['current_epoch']==0:
                                resume=False 
                                #In this case we will initialise. It will need to be initialised in the training run. We probably managed to shift over
                                #the last.pth but didn't even manage to finish a checkpoint. 
                            else:
                                #It may be possible that we exited early while writing a new best.pth file. In this case, we must use last.pth 
                                #as the best.pth as well because after all this is the only time it would have happened! 
                                # This is because we want to avoid corrupted checkpoints no matter what, so we will always delete best.pth before
                                #  writing a new one. 
                                resume = True 
                                shutil.copy(
                                    os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name, 'last.pth'),
                                    os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name, 'best.pth')
                                )
                                
                        else:
                            #We are definitely able to resume training because we have best.pth and last.pth both present. 
                            #So at least the initial validation must have been completed.
                            if stored_state['current_epoch'] == 0:
                                resume = False #We still need to start fresh until we can get one full epoch done.
                            else:
                                resume = True  
                else:
                    #We are definitely initialising. Lets delete the "best.pth" just in case it exists. 
                    resume = False
                    if os.path.exists(os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name, 'best.pth')):
                        #Shouldn't exist, lets delete it!
                        os.remove(os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name, 'best.pth'))
                    
                    existing_ckpt = torch.load(checkpoints.get(split_name), weights_only=False)

                    #Lets create the dir to store the checkpointing and save a starting point. 
                    os.makedirs(os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name), exist_ok=True)
    
                    #We use a tempdir so that if saving fails due to early-exit, it doesn't save a corrupted file and try to reload from that. Instead only
                    #when a fully written file is moved to the target location. 
                    # 
                    # Tempdir inception, one is temporary for training checkpoints, one is temporary
                    #in the conventional sense of temporary file writing.
                    
                    #This has now been converted to a function.

                    save_checkpoint(
                        temp_dir='storage_temp_kill_dir',
                        filename='last.pth',
                        desired_dir=os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name),
                        configs={
                            'current_epoch': 0, #We set epoch 0 as we are initialising training. This will ensure that the trainer will always train also,
                            #instead of loading the prior epoch state which would mean skiping training entirely in some cases.
                            'plans': adaptation_plans,
                            'prior_configs': {
                            'input_encoding': existing_ckpt['prior_configs']['input_encoding'],
                            'model_architecture': existing_ckpt['prior_configs']['model_architecture'],
                            'input_handling_configs': existing_ckpt['prior_configs']['input_handling_configs'],
                            'network_configuration': existing_ckpt['prior_configs']['network_configuration'],
                            'network_weights': existing_ckpt['prior_configs']['network_weights']}   
                            }
                    )
                        
                    #Why did we do it like this...(i.e. with saving checkpoints all over)? I don't know. Maybe I'll find a better explanation in me later
                    #other than "I was tired and wanted to finish this quickly". 
                        # Its easier to transfer model config and training info with the checkpointing this way :) we
                        #don't have to track a million variables explicitly I guess. 


                if self._callback is None:
                    raise RuntimeError("No callback configured; use configure_callback(cb, required_components=...) to set one")
                
                self.logger.info(f"Invoking adaptation callback for split {split_name} with resume={resume}")
                
                tensorboard_writer = SummaryWriter(
                    log_dir=os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name)
                )
                self._callback(
                    logger=self.logger,
                    tensorboard_writer=tensorboard_writer,
                    configs_labels_dict= configs_labels_dict,
                    tmp_dir=os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name),
                    split_name=split_name,
                    resume=resume,
                    dataloaders=dataloader_dict
                )
                #Closing the tensorboard writer to ensure all pending writes are done.
                tensorboard_writer.close() 
    
                #moving the tmp directory to a permanent one if the training completed successfully.
                shutil.move(
                    os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name), 
                    os.path.join(self.training_completion_dir, f'adaptation_{adaptation_number}', f"completed_{split_name}")
                ) 
                #Then we delete the "last.pth" file as we have completed training and only retain "best". 
                #We use a greedy-like search (not exactly because we retain all BEST checkpoints across episodes)
                #but only retain that which was "best" within a given adaptation/training configuration. 
                
                os.remove(os.path.join(
                    os.path.join(self.training_completion_dir, f'adaptation_{adaptation_number}', f"completed_{split_name}"), 
                    'last.pth'
                ))
                if os.path.exists(os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name)):
                    shutil.rmtree(os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name))

            else:
                #We have already completed this split's training. 
                self.logger.info(f"Skipping adaptation for split {split_name} as completed training found.")
                
                #Just delete the temporary training dir if exists, we will have to re-create once we adapt again.
                if os.path.exists(os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name)):
                    shutil.rmtree(os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}', split_name))

            #Storing the best checkpoint path for this split.
            #We will also still need to return the best checkpoint path for this split on re-run!     
            best_ckpt_path = os.path.join(self.training_completion_dir, f'adaptation_{adaptation_number}', f"completed_{split_name}", "best.pth")
            if not os.path.exists(best_ckpt_path):
                raise RuntimeError(f"Best.pth not found after completing training for split {split_name}!")
            else:
                checkpoints[split_name] = best_ckpt_path


        #If we are done with all splits, we can delete the temporary training dir entirely. Why did we retain
        #the entire tmp dir during adaptation of all splits? Main reason: being able to inspect convergence
        #/training behaviour on each training episode if training actually takes some time when num_samples is fairly large! (i.e. during training!)   
        if os.path.exists(os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}')): #Needs to delete if exists. Might not exist if we 
            #finished adaptation and deleted but were not yet able to reach the point of returning to the validation executor. 
            shutil.rmtree(os.path.join(self.training_tmp_dir, f'adaptation_{adaptation_number}'))

        self.logger.info(f"Adaptation for adaptation number {adaptation_number} completed for all splits.")
        if len(dataloaders) == 1: 
            return {
                'checkpoints': checkpoints,
                'best_ckpt_name': 'fold_0'  #Hardcoded for now 
            }
        else:
            raise Exception("Multi-split adaptation outcome handling not yet implemented for determining what to carry forward for inference.")