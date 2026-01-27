# i am your ghost hiii ;)  hi :), it sucks being here yeah the office is so empty,  no one from our lab is in also. call me if u feel lonely aite [redacted]] is here so dont want to disturb her, so its not fully empty then ha hano i guess not, im jk, yeah its annoying to not be able to go for a walk yeah also that. will probably leav e early tday. yeah me too, cya later then, bye bye :)  input_image.shape[1], input_image.shape[2], input_image.shape[3], input_image.shape[4]
#are you going to get that jumper today? yeah i think so, its quite nice. ok cool, see you later then, bye :) lmao the autocomplete is kind of funny. i am not sure about the jumper but maybe yh i
#shoudl go ahead, i think it's supposed to snow any minute now idk about [redacted]] or wherever u live, im in [redacted] that' s what i meant yes
#snow day without being around anyone wow thats just perfect. perfect day for some introspective looking out of the window time
#ive done TOO much of that recently hahahah ok then work!!!! noo :(, send pics if you go get the jumper though okay will do. i have that meeting in an hour so gonna go do some prep.
#good luck im rooting for you always thank you although it should be fine, she's just another phd student although shes ridiculously smart b
#but yeah should be fine. well you get a bit anxious for team meeting presentations so yea just trying to be supportive, thank you thank you i appreciate it. okay have fun with work see ya laterz. ;) see ya 

# wait what, take care of yourself and stay warm :) ty, will do. gym tomorrow? yeah im sleeping like  ababy tonig hlegs?l, hmm if you are i can yeah i am but nws if u want to do smth else. its either pull or legs for me aite see u there 7:30 sharp. lets see if the trains are running first :'(l)o'ololol aint shit funny hahahahha  !!!!!!! dude i was telling my cousin i was trying to move for that exact reason like 2 mins before i found out tdhat they were cancelled damn the universe is telling u to get out of hillingdon FR, gonn start looking this weekend.nice lmk if u need help im good at that stuff, do you have a good vibe check for houses hugely kk ty aite c ya send pics!! oki take care nyeee byebye



# This entire codebase.................thanks fabian.
# https://i.kym-cdn.com/entries/icons/original/000/034/623/Untitled-3.png


from abc import abstractmethod
import torch
import numpy as np
import gc 
import torch.nn as nn
import copy
import os
import torch.distributed as dist
from torch.cuda import amp
from typing import Any, Dict
import shutil 
from torch.utils.data import RandomSampler
#Imports for the components needed for training. 
from nnInteractive.adaptation.training_utils.network_configs import network_registry
from nnInteractive.adaptation.training_utils.loss_configs import loss_registry
from nnInteractive.adaptation.training_utils.optimisation_configs import optimiser_algo_registry as optimiser_registry
from nnInteractive.adaptation.training_utils.optimisation_configs import lr_scheduler_registry 
from nnInteractive.adaptation.training_utils.metric_configs import metric_registry
from nnInteractive.adaptation.training_utils.input_encoding import input_encoding_registry 
from nnInteractive.adaptation.training_utils.build_simple_heuristic import BuildHeuristic  
from nnInteractive.adaptation.training_utils.saving_utils import save_checkpoint
from nnInteractive.adaptation.data_handling import IndexedSampler
from monai.data import DataLoader 
class Trainer:
    def __init__(
        self,
        planner_config, 
        device: torch.device,
        ):
        self.planner_config = planner_config
        if device is torch.device('cpu'):
            raise RuntimeError("Trainer cannot be initialised on CPU device! Training requires GPU acceleration.")
        self.device = device
        self.preview_num_samples = 8  #Number of samples to preview for resume verification.
        self.epoch_saving_period = 1 #Hardcoding this for now, lets be extremely conservative with saving, writing 
        #should be relatively insignificant compared to training time within the epoch....... 
        self.network = None #This will be setup later.
        self.global_iter_step = 0 #This is the global step, primarily to be used for tensorboard logging.
        #Initialised with 0, will be updated on resume.
        self.ema_param = 0.9 #EMA parameter for best model tracking. 
        
    def tensorboard_writer_fn(self, value_dict: dict, step: int):
        for name, value in value_dict.items():
                self.tensorboard_writer.add_scalar(
                    name, value, step
                    )
    def forward_train(self, input_image, input_prompts, input_prompts_lbs, prev_pred = None, initialise=False, propagated_preds=None):
        self.network.train()
        input = self.construct_input(
            input_image, 
            input_prompts,
            input_prompts_lbs,
            prev_pred=prev_pred,
            initialise=initialise,
            propagated_preds=propagated_preds
            )
        with torch.cuda.amp.autocast():
            #Implement the forward pass here.
            output=self.network(input)

        input = None #Free up memory
        torch.cuda.empty_cache() #This is getting a bit annoying.......
        return output 
    
    def filter_finished_samples(
        self,
        finished_samples:tuple,
        output:torch.Tensor, 
        gt:torch.Tensor, 
        image:torch.Tensor,
        propagated_preds:dict, 
        batchwise_final_pred:torch.Tensor
        ):
        #Implement the finished sample filtering here.


        #For loops always slow but MVP! Lets just make sure its readable and correct for now.
        for fs in finished_samples.tolist():
            original_idx = propagated_preds[fs]
            batchwise_final_pred[original_idx:original_idx+1, :, :, :, :] = output.argmax(axis=1).unsqueeze(axis=1)[fs:fs+1, :, :, :, :]
            #Here we place the pred into the original index location.
            
            #Now we remove this sample from the propagated preds dict, according to the current batch index
            propagated_preds.pop(fs)
        #Now we have finished iterating, we can discard finished samples without worrying
        #about things messing up in-place etc.

        #Now let us update the gt, pred and image tensors to remove the finished samples.
        #NOTE: We must do it like this otherwise constantly re-indexing will make things confused.

        #We will do it by only retaining the non-finished samples. 
        gt = gt[torch.tensor(list(propagated_preds.keys())), :, :, :, :]
        image = image[torch.tensor(list(propagated_preds.keys())), :, :, :, :]
        output= output[torch.tensor(list(propagated_preds.keys())), :, :, :, :]

        #update the indices 
        propagated_preds = {new_idx: original_idx for new_idx, (_, original_idx) in enumerate(propagated_preds.items())}
    
        torch.cuda.empty_cache() #This is getting a bit annoying.......
        return output, gt, image, propagated_preds, batchwise_final_pred

    @torch.inference_mode()
    def forward_eval(self, input_image, input_prompts, input_prompts_lbs, prev_pred = None, initialise=False, propagated_preds=None):
        self.network.eval()
        input = self.construct_input(
            input_image, 
            input_prompts, 
            input_prompts_lbs,
            prev_pred,
            initialise=initialise,
            propagated_preds=propagated_preds)
        with torch.cuda.amp.autocast():
            #Implement the forward pass here.
            output=self.network(input)

        input = None #Free up memory
        torch.cuda.empty_cache() #This is getting a bit annoying.......
        return output 
    
    def construct_input(self, input_image, input_prompts, input_prompts_lbs, prev_pred, initialise=False, propagated_preds=None):
        #Implement the input construction here.
        #batch, Channels, height, width, depth
        b = input_image.shape[0]  
        c = self.interaction_encoder.num_interaction_channels + 2
        h,w,d = input_image.shape[2], input_image.shape[3], input_image.shape[4]
        assert [h,w,d] == self.input_handling_configs['patch_size'], "Input image spatial dimensions do not match expected patch size from input handling configs!"
        
        if initialise:
            assert propagated_preds == None, "Propagated preds must be None during initialisation! What is there to propagate yet!?"
            assert prev_pred == None
            self.input_buffer = torch.zeros((b,c,h,w,d), device=self.device, dtype=input_image.dtype)
            self.input_buffer[:,0:1,:,:,:] = input_image
            #Never mind, why the hell am i normalising here? This should be done in the dataloader so that
            #we can have a training pipeline which will actually allow for augmentation properly. 

            # #Normalise!!! DO NOT FORGET TO NORMALISE THE IMAGE INPUT!!! Can't believe we forgot to do this...
            # self.input_buffer[:,0:1,:,:,:] -= self.input_buffer[:,0:1,:,:,:].mean()
            # self.input_buffer[:,0:1,:,:,:] /= self.input_buffer[:,0:1,:,:,:].std()
        else:
            #Lets filter the input buffer according to the propagated preds if provided.
            if propagated_preds is not None:
                self.input_buffer = self.input_buffer[torch.tensor(list(propagated_preds.keys())), :, :, :, :]            
            assert prev_pred.shape[0] == b
            assert prev_pred.shape[2:] == (h,w,d)
            assert self.input_buffer.shape[0] == b, "Input buffer batch size does not match current batch size derived from input image and prompts"
           #We must have an existing buffer for editing.
            self.input_buffer[:, 1:2, :, :, :] = prev_pred

        #Filling in the interactions.
        self.input_buffer[:, 2:, :, :, :] = self.interaction_encoder(input_prompts, input_prompts_lbs, initialise, self.input_buffer[:, 2:, :, :, :])
        return self.input_buffer
    
    def train(self, epoch_num):
        self.logger.info('Starting training at epoch {}'.format(epoch_num))
        
        metric_dict = dict()

        for batch_idx in range(self.train_dataloading_config['total_iterations']):
            batch_data = next(iter(self.train_dataloader))
            assert batch_data['image'].shape[2:] == batch_data['label'].shape[2:], "Image and label spatial dimensions do not match!"
            batch_data = {k: v.cpu() for k, v in batch_data.items()}
            torch.cuda.empty_cache() #lets empty cache like a madman... 
            # oh if only we were blessed with 1TB of VRAM. 
            metric_dict[batch_idx] = dict() 

            self.global_iter_step += 1
            #Implement the training step here. 
            loss_dict = dict()
            per_iter_metric_dict = dict() #Initialise the metric dict for this batch which stores per-iter metrics.

            #First we simulate a prompt synthesis step (or lackthereof) to create the input to the network. 

            #First an initialisation. #NOTE: Future work may make it so that we can dynamically change the distribution of the
            #initialisation mechanism during training. 

            # #IF there is an empty sample then we will need to filter it. 
            # empty_samples = (batch_data['label'].sum(dim=[1,2,3,4]) == 0).nonzero(as_tuple=True)[0]
            # if len(empty_samples) > 0:
            #     if len(empty_samples) == batch_data['gt'].shape[0]:
            #         #In this case all samples were empty, we skip this batch.
            #         self.logger.warning("All samples in the current batch are empty! Skipping this batch during training.")
            #         continue
            #     else:
            #         #We have some empty samples but not all were empty, we need to filter them out.
            #         non_empty_samples = [i for i in range(batch_data['gt'].shape[0]) if i not in empty_samples.tolist()]
            #         for key in batch_data.keys():
            #             batch_data[key] = batch_data[key][non_empty_samples]
            
            #Now we have filtered out empty samples a-priori.
            gt = batch_data['label'].to(dtype=torch.int8) #.to(device=self.device, dtype=torch.int8)
            image = batch_data['image']#.to(device=self.device)
            
            if 'Interactive Init' in self.train_prompters.keys():
                input_prompts, input_prompts_lbs = self.train_prompters['Interactive Init'](
                    data={
                        'gt': gt.to(device=self.device).squeeze(axis=1), #We will squeeze the channel dimension for prompt gen.
                        'prev_pred': None
                    }
                )
                init = 'Interactive Init'
            elif 'Automatic Init' in self.train_prompters.keys():
                init = 'Automatic Init'
                raise NotImplementedError('Automatic Init not implemented yet in trainer!')
            else:
                raise RuntimeError("No valid initialisation mechanism found in train prompters!")
            torch.cuda.empty_cache() #This is getting a bit annoying.......
            output = self.forward_train(
                input_image=image.to(device=self.device), 
                input_prompts=input_prompts, 
                input_prompts_lbs=input_prompts_lbs,
                prev_pred=None,
                initialise=True
                )
            
            #OUTPUT is going to be logits with shape BCHWD where C = number of classes. (2 for binary semantic seg.)
            loss = self.calc_base_loss(
                output=output,
                target=gt.to(device=self.device)
                )
            torch.cuda.empty_cache() #This is getting a bit annoying.......

            #Now lets calculate the loss for the initial prediction.
            loss_dict.update(
                {
                init: {idx: val for idx, val in loss.items()} #Initially, the correspondence is
                #fully 1-to-1 between original and current batch indices.
                }
            )
            #Now we detach the output to avoid gradient flow through the interaction loop. 
            output = output.clone().detach()

            per_iter_metric_dict.setdefault('init', dict())
            #Initialise the per-iter metrics for the initial iteration. 
            for metric, metric_fn in self.per_iter_train_metrics.items():
                if self.per_iter_train_metric_wrappers[metric]['iteration'] == 'all':
                    per_iter_metric_dict['init'][metric] = dict()
                    vals = metric_fn(
                        output.argmax(axis=1).unsqueeze(axis=1), 
                        gt.to(device=self.device),
                        self.configs_labels_dict
                        )
                    #Now lets distribute them into a batch separated dict. 
                    for idx in range(vals.shape[0]):
                        per_iter_metric_dict['init'][metric][idx] = vals[idx].cpu()       

                torch.cuda.empty_cache() #This is getting a bit annoying.......

            batchwise_final_pred = output.argmax(axis=1).unsqueeze(axis=1) #Put back the channel dimension.
            #We store a batchwise pred so that we have the "final" pred of a given batch sample at the end of the
            # interaction loop. This is because there may be inconsistencies in early-exit behaviour.
            
            #We store a list of  preds batch indices so that we know which samples are still active, and what it
            # corresponds to in the original batch here.
             
            propagated_preds = {i:i for i in 
                list(range(gt.shape[0])) #Initially, all remaining samples are active, their indices
                #correspond to the same index. 
                #KEY = current index in batch_data, VALUE = original index in batch.
            }

            #Lets check if can do any early exits right away.
            finished_samples = (torch.sum((output.argmax(axis=1).unsqueeze(axis=1) == gt.to(device=self.device)).int(), dim=[1,2,3,4]) == gt.shape[2]*gt.shape[3]*gt.shape[4]).nonzero(as_tuple=True)[0].cpu()
            if len(finished_samples) > 0:
                output, gt, image, propagated_preds, batchwise_final_pred = self.filter_finished_samples(
                        finished_samples,
                        output, 
                        gt, 
                        image, 
                        propagated_preds, 
                        batchwise_final_pred        
                ) 
            if len(propagated_preds) != 0: #Now we enter the editing loops if there are still samples to edit.
                for i in range(self.train_inner_loop_conf['max_interactions']):
                    
                    input_prompts, input_prompts_lbs = self.train_prompters['Interactive Edit'](
                        data={
                            'gt': gt.squeeze(axis=1).to(device=self.device), #We will squeeze the channel dimension.
                            'prev_pred': output.argmax(axis=1) #Axis=1 is the class dimension. 
                        }
                    )

                    torch.cuda.empty_cache() #This is getting a bit annoying.......
                    
                    output = self.forward_train(
                    input_image=image.to(device=self.device), 
                    input_prompts=input_prompts, 
                    input_prompts_lbs=input_prompts_lbs,
                    prev_pred=output.argmax(axis=1).unsqueeze(axis=1), #Put back the channel dimension.
                    initialise=False,
                    propagated_preds=propagated_preds)
            
                    #OUTPUT is going to be logits with shape BCHWD where C = number of classes. (2 for binary semantic seg.)
                    loss = self.calc_base_loss(
                        output=output,
                        target=gt.to(device=self.device)
                        )
                    torch.cuda.empty_cache() #This is getting a bit annoying.......

                    #Now lets calculate the loss for the initial prediction.
                    loss_dict.update(
                        { #We reindex according to the propagated preds original indices.
                        f'Interactive Edit Iter {i}': {orig_idx: loss[current_idx] for current_idx, orig_idx in propagated_preds.items()}
                        }
                    )
                    output = output.clone().detach()

                    #Lets calculate any metrics depending on the wrapper arguments:
                    per_iter_metric_dict[f'Interactive Edit Iter {i}'] = dict()

                    for metric, metric_fn in self.per_iter_train_metrics.items():
                        if self.per_iter_train_metric_wrappers[metric]['iteration'] == 'all':
                            per_iter_metric_dict[f'Interactive Edit Iter {i}'][metric] = dict()
                            vals = metric_fn(
                                output.argmax(axis=1).unsqueeze(axis=1), 
                                gt.to(device=self.device),
                                self.configs_labels_dict
                                )
                            #Now lets distribute them into a batch separated dict. 
                            for current, orig in propagated_preds.items():
                                per_iter_metric_dict[f'Interactive Edit Iter {i}'][metric][orig] = vals[current].cpu()      
                        torch.cuda.empty_cache() #This is getting a bit annoying.......

                    #Here we check if we should early exit, we terminate a sample if the segmentation perfectly aligns.
                    
                    #IF there is an finished sample then we will need to filter it. 
                    finished_samples = (torch.sum((output.argmax(axis=1).unsqueeze(axis=1) == gt.to(device=self.device)).int(), dim=[1,2,3,4]) == gt.shape[2]*gt.shape[3]*gt.shape[4]).nonzero(as_tuple=True)[0].cpu()
                    #Now we need to index into the batchwise final pred to place the finished samples.
                    if len(finished_samples) > 0:
                        output, gt, image, propagated_preds, batchwise_final_pred = self.filter_finished_samples(
                            finished_samples,
                            output, gt, image, propagated_preds, batchwise_final_pred
                        )
                        
                    else:
                        pass #No finished samples, do nothing. 
                    
                    if len(propagated_preds) == 0:
                        #All samples have finished, we can break out of the interaction loop.
                        break
                
            else:
                pass #Samples are all finished, so we can't loop, just calculate the metric.

            #Call on a function which updates the parameters based on the loss dict. This is effectively the 
            #wrapper around the base loss, and dictates how we should update the parameters based on the unrolled
            #set of interaction states. 
            self.update_parameters(loss_dict, batch_size=batch_data['label'].shape[0])
            #Here we use the batch size which corresponds to the a-priori filtered empty samples. NOT the 
            #gt which can be updated during early-exit, as the a-priori filtering has necessitated that
            #at least the first inference callback is always run for all samples in the batch.

            #This is for metrics which are calculated per iteration. 

            #NOTE: We expect that the outputted metrics be a batchlength tensor of values for each metric.    
            per_iter_metric_dict.setdefault('final', dict())
            for metric, metric_fn in self.per_iter_train_metrics.items():
                if self.per_iter_train_metric_wrappers[metric]['iteration'] == 'final':
                    per_iter_metric_dict['final'][metric] = dict()
                    vals = metric_fn(
                        batchwise_final_pred, 
                        batch_data['label'].to(device=self.device), #Calc with the original gt!
                        self.configs_labels_dict
                        )
                    #Now lets distribute them into a batch separated dict. 
                    for idx in range(vals.shape[0]):
                        per_iter_metric_dict['final'][metric][idx] = vals[idx].cpu()
                torch.cuda.empty_cache() #This is getting a bit annoying.......

            #Now lets move them into the batch level metrics dict.

            #Now lets move the metrics from this batch into the overall metric dict.
            metric_dict[batch_idx]['per_iter'] = per_iter_metric_dict

            #Now lets calculate the aggregate metrics, we pass in the per-iter metrics dict for this.

            #First we will check that we have all of the required per-iter metrics (where possible to be handled)
            for metric in self.aggregate_train_metrics.keys():
                if self.aggregate_train_metric_wrappers[metric]['iteration'] == 'all':
                    #Assert all of the iterations are present in the metric dict! 
                    assert all([self.aggregate_train_metric_wrappers[metric]['base_metric'] in per_iter_metric_dict[iter_key].keys() for iter_key in per_iter_metric_dict.keys() if iter_key != 'final']), f"Not all iterations found in per-iter metric dict for batch {batch_idx} when required for aggregate metric {metric}!"
                elif self.aggregate_train_metric_wrappers[metric]['iteration'] == 'final':
                    assert self.aggregate_train_metric_wrappers[metric]['base_metric'] in per_iter_metric_dict['final'], f"Final iteration not found in per-iter metric dict for batch {batch_idx} when required for aggregate metric {metric}!"

            metric_dict[batch_idx]['aggregate'] = {
                metric: metric_fn(
                    per_iter_metric_dict,
                    self.aggregate_train_metric_wrappers[metric],
                    batch_data['label'].shape[0]) #Batch size)
                for metric, metric_fn in self.aggregate_train_metrics.items()
            }
            torch.cuda.empty_cache() #This is getting a bit annoying.......

        #Now that we have finished all batches for this epoch, we can do the writing of metrics to tensorboard.
        write_per_iter_metrics = {metric: None for metric in self.per_iter_train_metrics.keys() if self.per_iter_train_metric_wrappers[metric]['write']}
        for metric in write_per_iter_metrics.keys(): 
            #Lets gather the list of values across all batches for an epoch summarised metric.
            if not self.per_iter_train_metric_wrappers[metric]['write']:
                raise RuntimeError(f"Metric {metric} is not configured to be written, cannot gather values for writing!")
            
            if self.per_iter_train_metric_wrappers[metric]['iteration'] == 'final':
                if [metric_dict[batch_idx]['per_iter']['final'][metric] for batch_idx in metric_dict.keys()] == []:
                    raise Exception('Metric dict is fully empty, please configure the dataloader so that it is' \
                    'not possible for all batches in an epoch to be empty!')
                else:
                    write_per_iter_metrics[metric] = torch.tensor(
                        [list(metric_dict[batch_idx]['per_iter']['final'][metric].values()) for batch_idx in metric_dict.keys()]).mean() #Average over all samples (batch and epoch!)
                    
            elif self.per_iter_val_metric_wrappers[metric]['iteration'] == 'all':
                raise NotImplementedError('Not supported to aggregate per iteration metric scores for writing metrics, this is left to the aggregate metrics.')
            else:
                raise NotImplementedError("Unknown iteration type in val metric wrapper!")
        
        #Now lets do the aggregation for the aggregate metrics. 
        write_aggregate_metrics = {metric: None for metric in self.aggregate_train_metrics.keys() if self.aggregate_train_metric_wrappers[metric]['write']}
        for metric in write_aggregate_metrics.keys():
            if not self.aggregate_train_metric_wrappers[metric]['write']:
                raise RuntimeError(f"Metric {metric} is not configured to be written, cannot gather values for writing!")
            #Lets gather the list of values across all batches for an epoch summarised metric.
            if [metric_dict[batch_idx]['aggregate'][metric] for batch_idx in metric_dict.keys()] == []:
                raise Exception('Metric dict is fully empty, please configure the dataloader so that it is not')
            else:
                write_aggregate_metrics[metric] = torch.tensor(
                    [list(metric_dict[batch_idx]['aggregate'][metric].values()) for batch_idx in metric_dict.keys()]).mean() #Average over all samples (batch and epoch!)

        self.tensorboard_writer_fn(write_per_iter_metrics, epoch_num)
        self.tensorboard_writer_fn(write_aggregate_metrics, epoch_num)
        
        if epoch_num == 1:
            self.train_metric_history = {
            epoch_num: {'per_iter': write_per_iter_metrics, 'aggregate': write_aggregate_metrics}
        }
        else: 
            self.train_metric_history.update({
            epoch_num: {'per_iter': write_per_iter_metrics, 'aggregate': write_aggregate_metrics}
        })

        # self.logger.info("Train Mean Dice - " + str(metric_dict['Train Mean Dice']))
        #one last time.

        #Lets try and flush the cache, not fast but as long as our code doesn't break thats more important.
        del image 
        del gt 
        del output 
        del batch_data
        del loss_dict
        del per_iter_metric_dict
        torch.cuda.empty_cache() #This is getting a bit annoying.......

    def update_parameters(
        self, 
        loss_dict: dict,
        batch_size: int):
        #Selected losses to merge:
        if self.loss_wrapper_conf['used_outputs'] == 'all': #Use all per-iter outputs.
            used_losses = {
                i: [loss[i] for loss in loss_dict.values() if i in loss.keys()] for i in range(batch_size)
            }
        else:
            raise NotImplementedError("Only 'all' used_outputs supported currently in loss wrapper!")
        
        #Padding strategy: 
        if self.loss_wrapper_conf['early_exit_padding_strategy'] == 'Unpadded':
            pass #Do nothing.
        else:
            raise NotImplementedError("Only 'Unpadded' early exit padding strategy supported currently in loss wrapper!")
        
        #Merge strategy:
        if self.loss_wrapper_conf['merge_strategy'] == 'sum':
            # total_loss = sum(used_losses)
            raise NotImplementedError("Sum merge strategy not implemented yet!")
        elif self.loss_wrapper_conf['merge_strategy'] == 'mean':
            #Lets take the mean on a per-sample basis, and then 
            total_loss = torch.mean(
                torch.stack(
                    [torch.mean(torch.stack(used_losses[i])) for i in range(batch_size)]
                    )
                )
        total_loss = total_loss.to(device=self.device) #Lets move this to the correct device for the update.
        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.step(self.optimiser)
        self.grad_scaler.update()
        #Reset the gradients to avoid accumulation from previous step.
        self.optimiser.zero_grad() 
        self.tensorboard_writer_fn({
            self.base_loss.name: total_loss
        }, self.global_iter_step) 

        del total_loss
        torch.cuda.empty_cache() #This is getting a bit annoying.......

    def validate(self, epoch_num):
        self.logger.info('Starting validation at epoch {}'.format(epoch_num))
        
        metric_dict = dict() 
        
        for batch_idx in range(self.val_dataloading_config['total_iterations']):
            batch_data = next(iter(self.val_dataloader))
            assert batch_data['image'].shape[2:] == batch_data['label'].shape[2:], "Image and label spatial dimensions do not match!"
            batch_data = {k: v.cpu() for k, v in batch_data.items()}
            torch.cuda.empty_cache()
            metric_dict[batch_idx] = dict()    
            per_iter_metric_dict = dict() #Initialise the metric dict for this batch which stores per-iter metrics.

            #First we simulate a prompt synthesis step (or lackthereof) to create the input to the network. 

            #First an initialisation. #NOTE: Future work may make it so that we can dynamically change the distribution of the
            #initialisation mechanism during training. 

            #IF there is an empty sample then we will need to filter it. 
            # empty_samples = (batch_data['label'].sum(dim=[1,2,3,4]) == 0).nonzero(as_tuple=True)[0]
            # if len(empty_samples) > 0:
            #     if len(empty_samples) == batch_data['gt'].shape[0]:
            #         #In this case all samples were empty, we skip this batch.
            #         self.logger.warning("All samples in the current batch are empty! Skipping this batch during training.")
            #         continue
            #     else:
            #         #We have some empty samples but not all were empty, we need to filter them out.
            #         non_empty_samples = [i for i in range(batch_data['gt'].shape[0]) if i not in empty_samples.tolist()]
            #         for key in batch_data.keys():
            #             batch_data[key] = batch_data[key][non_empty_samples]
            
            #Now we have filtered out empty samples a-priori.
            gt = batch_data['label'].to(dtype=torch.int8)
            image = batch_data['image']#.to(device=self.device)
            
            if 'Interactive Init' in self.val_prompters.keys():
                input_prompts, input_prompts_lbs = self.val_prompters['Interactive Init'](
                    data={
                        'gt': gt.squeeze(axis=1).to(device=self.device), #We will squeeze the channel dimension for prompt gen.
                        'prev_pred': None
                    }
                )
                init = 'Interactive Init'
            elif 'Automatic Init' in self.val_prompters.keys():
                init = 'Automatic Init'
                raise NotImplementedError('Automatic Init not implemented yet in trainer!')
            else:
                raise RuntimeError("No valid initialisation mechanism found in train prompters!")
            
            output = self.forward_eval(
                input_image=image.to(device=self.device), 
                input_prompts=input_prompts, 
                input_prompts_lbs=input_prompts_lbs,
                prev_pred=None,
                initialise=True
                )
            torch.cuda.empty_cache() #This is getting a bit annoying.......

            #OUTPUT is going to be logits with shape BCHWD where C = number of classes. (2 for binary semantic seg.)
            per_iter_metric_dict.setdefault('init', dict())
            
            for metric, metric_fn in self.per_iter_val_metrics.items():
                if self.per_iter_val_metric_wrappers[metric]['iteration'] == 'all':
                    per_iter_metric_dict['init'][metric] = dict()
                    vals = metric_fn(
                        output.argmax(axis=1).unsqueeze(axis=1), 
                        gt.to(device=self.device),
                        self.configs_labels_dict
                        )
                    #Now lets distribute them into a batch separated dict. 
                    for idx in range(vals.shape[0]):
                        per_iter_metric_dict['init'][metric][idx] = vals[idx].cpu()       
                torch.cuda.empty_cache() #This is getting a bit annoying.......       


            batchwise_final_pred = output.argmax(axis=1).unsqueeze(axis=1) #Put back the channel dimension.
            #We store a batchwise pred so that we have the "final" pred of a given batch sample at the end of the
            # interaction loop. This is because there may be inconsistencies in early-exit behaviour.
            
            #We store a list of  preds batch indices so that we know which samples are still active, and what it
            # corresponds to in the original batch here.
             
            propagated_preds = {i:i for i in 
                list(range(gt.shape[0])) #Initially, all remaining samples are active, their indices
                #correspond to the same index. 
                #KEY = current index in batch_data, VALUE = original index in batch.
            }

            #Lets check if can do any early exits right away.
            finished_samples = (torch.sum((output.argmax(axis=1).unsqueeze(axis=1) == gt.to(device=self.device)).int(), dim=[1,2,3,4]) == gt.shape[2]*gt.shape[3]*gt.shape[4]).nonzero(as_tuple=True)[0]
            if len(finished_samples) > 0:
                output, gt, image, propagated_preds, batchwise_final_pred = self.filter_finished_samples(
                        finished_samples,
                        output, 
                        gt, 
                        image, 
                        propagated_preds, 
                        batchwise_final_pred        
                ) 
            if len(propagated_preds) != 0: #Now we enter the editing loops if there are still samples to edit.
                for i in range(self.val_inner_loop_conf['max_interactions']):
                    
                    input_prompts, input_prompts_lbs = self.val_prompters['Interactive Edit'](
                        data={
                            'gt': gt.to(device=self.device).squeeze(axis=1), #We will squeeze the channel dimension.
                            'prev_pred': output.argmax(axis=1).to(device=self.device) #Axis=1 is the class dimension. 
                        }
                    )
                    output = self.forward_eval(
                    input_image=image.to(device=self.device), 
                    input_prompts=input_prompts, 
                    input_prompts_lbs=input_prompts_lbs,
                    prev_pred=output.argmax(axis=1).unsqueeze(axis=1).to(device=self.device), #Put back the channel dimension.
                    initialise=False,
                    propagated_preds=propagated_preds)
            
                    #OUTPUT is going to be logits with shape BCHWD where C = number of classes. (2 for binary semantic seg.)
                    torch.cuda.empty_cache() #This is getting a bit annoying.......

                    #Lets calculate any metrics depending on the wrapper arguments:
                    per_iter_metric_dict[f'Interactive Edit Iter {i}'] = dict()

                    for metric, metric_fn in self.per_iter_val_metrics.items():
                        if self.per_iter_val_metric_wrappers[metric]['iteration'] == 'all':
                            per_iter_metric_dict[f'Interactive Edit Iter {i}'][metric] = dict()
                            vals = metric_fn(
                                output.argmax(axis=1).unsqueeze(axis=1), 
                                gt.to(device=self.device),
                                self.configs_labels_dict
                                )
                            #Now lets distribute them into a batch separated dict. 
                            for current, orig in propagated_preds.items():
                                per_iter_metric_dict[f'Interactive Edit Iter {i}'][metric][orig] = vals[current].cpu()       
                    torch.cuda.empty_cache() #This is getting a bit annoying.......

                    #Here we check if we should early exit, we terminate a sample if the segmentation perfectly aligns.
                    
                    #IF there is a finished sample then we will need to filter it. 
                    finished_samples = (torch.sum((output.argmax(axis=1).unsqueeze(axis=1) == gt.to(device=self.device)).int(), dim=[1,2,3,4]) == gt.shape[2]*gt.shape[3]*gt.shape[4]).nonzero(as_tuple=True)[0]
                    #Now we need to index into the batchwise final pred to place the finished samples.
                    if len(finished_samples) > 0:
                        output, gt, image, propagated_preds, batchwise_final_pred = self.filter_finished_samples(
                            finished_samples,
                            output, gt, image, propagated_preds, batchwise_final_pred
                        )
                        
                    else:
                        pass #No finished samples, do nothing. 
                    
                    if len(propagated_preds) == 0:
                        #All samples have finished, we can break out of the interaction loop.
                        break
                
            else:
                pass #Samples are all finished, so we can't loop, just calculate the metric.

            #This is for metrics which are calculated per iteration. 

            #NOTE: We expect that the outputted metrics be a batchlength tensor of values for each metric.    
            per_iter_metric_dict.setdefault('final', dict())
            for metric, metric_fn in self.per_iter_val_metrics.items():
                if self.per_iter_val_metric_wrappers[metric]['iteration'] == 'final':
                    per_iter_metric_dict['final'][metric] = dict()
                    vals = metric_fn(
                        batchwise_final_pred, 
                        batch_data['label'].to(device=self.device), #Calc with the original gt!
                        self.configs_labels_dict
                        ).cpu()
                    #Now lets distribute them into a batch separated dict. 
                    for idx in range(vals.shape[0]):
                        per_iter_metric_dict['final'][metric][idx] = vals[idx].cpu()
                    torch.cuda.empty_cache() #This is getting a bit annoying.......
            #Now lets move them into the batch level metrics dict.

            #Now lets move the metrics from this batch into the overall metric dict.
            metric_dict[batch_idx]['per_iter'] = per_iter_metric_dict

            #Now lets calculate the aggregate metrics, we pass in the per-iter metrics dict for this.

            #First we will check that we have all of the required per-iter metrics (where possible to be handled)
            for metric in self.aggregate_val_metrics.keys():
                if self.aggregate_val_metric_wrappers[metric]['iteration'] == 'all':
                    #Assert all of the iterations are present in the metric dict! 
                    assert all([self.aggregate_val_metric_wrappers[metric]['base_metric'] in per_iter_metric_dict[iter_key].keys() for iter_key in per_iter_metric_dict.keys() if iter_key != 'final']), f"Not all iterations found in per-iter metric dict for batch {batch_idx} when required for aggregate metric {metric}!"
                elif self.aggregate_val_metric_wrappers[metric]['iteration'] == 'final':
                    assert self.aggregate_val_metric_wrappers[metric]['base_metric'] in per_iter_metric_dict['final'], f"Final iteration not found in per-iter metric dict for batch {batch_idx} when required for aggregate metric {metric}!"

            metric_dict[batch_idx]['aggregate'] = {
                metric: metric_fn(
                    per_iter_metric_dict,
                    self.aggregate_val_metric_wrappers[metric],
                    batch_data['label'].shape[0]) #Batch size)
                for metric, metric_fn in self.aggregate_val_metrics.items()
            }
        
        write_per_iter_metrics = {metric: None for metric in self.per_iter_val_metrics.keys() if self.per_iter_val_metric_wrappers[metric]['write']}
        for metric in write_per_iter_metrics.keys():
            #Lets gather the list of values across all batches for an epoch summarised metric.
            if not self.per_iter_val_metric_wrappers[metric]['write']:
                raise RuntimeError(f"Metric {metric} is not configured to be written, cannot gather values for writing!")
            
            if self.per_iter_val_metric_wrappers[metric]['iteration'] == 'final':
                if [metric_dict[batch_idx]['per_iter']['final'][metric] for batch_idx in metric_dict.keys()] == []:
                    raise Exception('Metric dict is fully empty, please configure the dataloader so that it is' \
                    'not possible for all batches in an epoch to be empty!')
                else:
                    write_per_iter_metrics[metric] = torch.tensor(
                        [list(metric_dict[batch_idx]['per_iter']['final'][metric].values()) for batch_idx in metric_dict.keys()]).mean() #Average over all samples (batch and epoch!)
                    
            elif self.per_iter_val_metric_wrappers[metric]['iteration'] == 'all':
                raise NotImplementedError('Not supported to aggregate per iteration metric scores for writing metrics, this is left to the aggregate metrics.')
            else:
                raise NotImplementedError("Unknown iteration type in val metric wrapper!")
        
        #Now lets do the aggregation for the aggregate metrics. 
        write_aggregate_metrics = {metric: None for metric in self.aggregate_val_metrics.keys() if self.aggregate_val_metric_wrappers[metric]['write']}
        for metric in write_aggregate_metrics.keys():
            if not self.aggregate_val_metric_wrappers[metric]['write']:
                raise RuntimeError(f"Metric {metric} is not configured to be written, cannot gather values for writing!")
            #Lets gather the list of values across all batches for an epoch summarised metric.
            if [metric_dict[batch_idx]['aggregate'][metric] for batch_idx in metric_dict.keys()] == []:
                raise Exception('Metric dict is fully empty, please configure the dataloader so that it is not')
            else:
                write_aggregate_metrics[metric] = torch.tensor(
                    [list(metric_dict[batch_idx]['aggregate'][metric].values()) for batch_idx in metric_dict.keys()]).mean() #Average over all samples (batch and epoch!)

        #Dumping variables to free up GPU memory.
        del image 
        del gt
        del output 
        del batch_data 
        torch.cuda.empty_cache() #This is getting a bit annoying.......

        #Lets determine what the dict we need to pull from for comparison is.
        if self.improvement_criterion_config['metric_type'] == 'per_iter':
            comparison_dict = write_per_iter_metrics
        elif self.improvement_criterion_config['metric_type'] == 'aggregate':
            comparison_dict = write_aggregate_metrics
        else:
            raise RuntimeError(f"Unknown metric type {self.improvement_criterion_config['metric_type']} for improvement criterion!")
        
        criterion_metric = self.improvement_criterion_config['metric_name']
        assert criterion_metric in comparison_dict.keys(), f"Criterion metric {criterion_metric} not found in metrics dict keys!"
            
        #We pull the metric dictionary which has been aggregated over the epoch for comparison. 
        
        #We keep track of all validation metrics. This helps to calculate EMA performance etc.


        if epoch_num == 0:
            new_best = True #We always have a new best at epoch 0, it is our starting point!
            self.best_metrics = comparison_dict[criterion_metric]
            self.best_ema_metrics = comparison_dict[criterion_metric]
            self.val_metric_history = {
                epoch_num: comparison_dict[criterion_metric]
            }
            self.best_epoch = epoch_num
        else:
            new_best = False #Default to no new best.
            
            # assert criterion_metric in self.best_metrics and criterion_metric in self.best_ema_metrics, f"Criterion metric {criterion_metric} not found in best metrics or best ema metrics!"
            assert self.best_metrics is not None, "Best metrics is None at epoch > 0!"
            assert self.best_ema_metrics is not None, "Best EMA metrics is None at epoch > 0!"
            self.val_metric_history.update({
                epoch_num: comparison_dict[criterion_metric]
            })
            
            ema_metric = self.val_metric_history[epoch_num] * self.ema_param + (1 - self.ema_param) * self.val_metric_history[epoch_num - 1]
            if self.improvement_criterion_config['criterion'] == 'strictly_greater':
                if  ema_metric > self.best_ema_metrics:
                    self.best_ema_metrics = ema_metric #We store the EMA metric as the best metric, this is our point of comparison after all!
                    self.best_metrics = comparison_dict[criterion_metric]
                    self.best_epoch = epoch_num
                    new_best = True
            elif self.improvement_criterion_config['criterion'] == 'greater_equal':
                if ema_metric >= self.best_ema_metrics:
                    self.best_ema_metrics = ema_metric
                    self.best_metrics = comparison_dict[criterion_metric]
                    self.best_epoch = epoch_num
                    new_best = True
            else:
                raise RuntimeError(f"Unknown improvement criterion {self.improvement_criterion_config['criterion']}!")
            
        #Writing the metrics to tensorboard.
        self.tensorboard_writer_fn(write_per_iter_metrics, epoch_num)
        self.tensorboard_writer_fn(write_aggregate_metrics, epoch_num)
        
        if new_best:
            self.logger.info(f"New best model found at epoch {epoch_num} with {criterion_metric} = {comparison_dict[criterion_metric]}. ")
            self.logger.info(f"Best EMA {criterion_metric} is now {self.best_ema_metrics}. ")
        return new_best 

    def calc_base_loss(self, output, target):        
        return self.base_loss(output, target)

        
    
    def setup_training(self):
        self.setup_trainable_model() 
        train_plans = self.stored_state['plans']['train_handlers'][self.split_name] 
        if train_plans == None:
            raise RuntimeError(f"Train plans for split {self.split_name} is None! Cannot setup training.")
        self.train_dataloading_config = train_plans['dataloading_config']
        self.setup_loss(train_plans)
        self.setup_optimiser(train_plans)
        self.setup_lr_schedulers(train_plans)
        self.setup_prompter(train_plans, plan_type='train_handlers')
        self.setup_grad_scaler()
        if not self.resume:
            pass
        else:
            #In this case we resuming training, and so we need to load the optimiser state dict. 
            #NOTE: For now we are assuming a static LR scheduler, if not then we will need to adjust this so that it can cross-check the scheduler name!?
            
            #First it needs to load the lr scheduler state dict, otherwise the LR params will be wrong, see: 
            #https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html description of load_state_dict.

            # "Make sure this method is called after initializing torch.optim.lr_scheduler.LRScheduler, 
            # as calling it beforehand will overwrite the loaded learning rates."
            if self.stored_state.get('lr_scheduler_state_dict') is None:
                raise RuntimeError("No LR scheduler state dict found in stored state for resume!")
            self.current_lr_scheduler.load_state_dict(self.stored_state['lr_scheduler_state_dict'])

            if self.stored_state.get('optimiser_state_dict') is None:
                raise RuntimeError("No optimiser state dict found in stored state for resume!")
            self.optimiser.load_state_dict(self.stored_state['optimiser_state_dict']) 

            if self.stored_state.get('grad_scaler_state_dict') is None:
                raise RuntimeError("No grad scaler state dict found in stored state for resume!")
            self.grad_scaler.load_state_dict(self.stored_state['grad_scaler_state_dict'])

    def setup_validation(self):
        
        val_plans = self.stored_state['plans']['val_handlers'][self.split_name]
        if val_plans == None:
            raise RuntimeError(f"Validation plans for split {self.split_name} is None! Cannot setup validation.")
        self.val_dataloading_config = val_plans['dataloading_config'] 
        #We need to set up the following: 
        #1) The prompter for validation, 2) The metrics for validation/performance tracking 
        self.setup_prompter(val_plans, plan_type='val_handlers')
        self.setup_performance_tracking(val_plans) 
        
    def setup_prior_model(self):
        '''
        This is a function which is intended for initialising the prior model for performing the 
        validation epoch before training starts to get a baseline with the existing model. 
        '''
        #Need to store the input handling config as this will be used for forming inputs in the initial validation...
        self.input_handling_configs = self.stored_state['prior_configs']['input_handling_configs']
        if self.input_handling_configs == None or self.stored_state['prior_configs']['model_architecture'] == None or self.stored_state['prior_configs']['network_configuration'] == None or self.stored_state['prior_configs']['network_weights'] == None:
            raise RuntimeError("Prior model configs or weights are None when trying to setup prior model! Cannot proceed.") 
        #Now we initialise the class which will process the prompts into the structure expected for the network.
        self.interaction_encoder = input_encoding_registry[self.stored_state['prior_configs']['input_encoding']](
            {'input_handling_configs': self.input_handling_configs}
        )


        network_factory = network_registry[self.stored_state['prior_configs']['model_architecture']]
        network_builder_class = network_factory(
            {'existing_kwargs':None, 'current_kwargs':self.stored_state['prior_configs']['network_configuration']}
        )
        self.network = network_builder_class.build_network_architecture(device=self.device)
        self.network.load_state_dict(self.stored_state['prior_configs']['network_weights'])
        if self.network == None:
            raise RuntimeError("Prior model is None after setup! Cannot proceed.") 
        
        self.logger.info("Prior model setup complete.") 

    def setup_trainable_model(self):
       
        self.input_handling_configs = None #Initialising to None for safety.
        self.network = None #Initialising to None for safety.
        torch.cuda.empty_cache() 

        #Loading configurations from the planner!
        
        self.input_handling_configs = self.planner_config['algorithm_config']['input_handling_configs']
        self.network_configuration = self.planner_config['algorithm_config']['network_configuration']
        self.network_architecture = self.planner_config['algorithm_config']['model_architecture'] 

        #We need to load the prior configs too so we can see if there is a discrepancy when 
        # initialising the network. This is important, because we may be adapting a model, where existing weights etc form
        # part of the new model. And so, we need to check whether this is the case, as we may need to pass through the 
        # existing configuration information 
        prior_input_handling_configs = self.stored_state['prior_configs']['input_handling_configs']
        prior_network_architecture = self.stored_state['prior_configs']['model_architecture']
        prior_network_configuration = self.stored_state['prior_configs']['network_configuration'] 
        prior_network_weights = self.stored_state['prior_configs']['network_weights']
        if prior_input_handling_configs == None or prior_network_architecture == None or prior_network_configuration == None or prior_network_weights == None:
            raise RuntimeError("Prior model configs or weights are None when trying to setup trainable model! Cannot proceed.")
        #I know it seems counter intuitive, even if we are loading an interrupted run. But this is just nomenclature inherited from how we load from
        #prior episodes. Need to just be careful to ensure that these variables are updated before storing the state!!
        discrepancy = False
        if self.input_handling_configs != prior_input_handling_configs:
            discrepancy = True 
        if self.network_configuration != prior_network_configuration:
            discrepancy = True
        if self.network_architecture != prior_network_architecture:
            discrepancy = True
        #This discrepancy bool is used to determine whether we need to pass through prior config to the new initialiser!
         
        network_factory = network_registry[self.planner_config['algorithm_config']['model_architecture']]
        if discrepancy:
            #In this case there is a discrepancy, and so we need to pass through the existing configs
            #so that the discrepancy can be resolved in building the new network and loading the weights which are
            #being transferred.
            network_builder_class = network_factory(
                {'existing_kwargs':self.stored_state['prior_configs'], 'current_kwargs':self.planner_config['algorithm_config']['network_configuration']}
            )
            assert network_builder_class != None, "Network builder class is None despite discrepancy in model configs!"
            assert getattr(network_builder_class, "build_network_architecture") is not None, "Network builder class has no build_network_architecture method!"
            assert getattr(network_builder_class, "load_weights") is not None, "Network builder class has no load_weights method!"
            self.network = network_builder_class.build_network_architecture(device=self.device)
            self.network = network_builder_class.load_weights(
                device=self.device, 
                model=self.network, 
                network_weights=prior_network_weights
                )
        else:
            #In this case the checkpoint aligns, so we just need to load all of the weights. 
            network_builder_class = network_factory(
                {'existing_kwargs':None, 'current_kwargs':self.planner_config['algorithm_config']['network_configuration']}
            )
            self.network = network_builder_class.build_network_architecture(device=self.device)
            self.network = network_builder_class.load_weights(
                device=self.device, 
                model=self.network,
                network_weights=prior_network_weights
                )


        #Now we initialise the class which will process the prompts into the structure expected for the network.
        self.interaction_encoder = input_encoding_registry[self.planner_config['algorithm_config']['input_encoding']](
            {'input_handling_configs': self.input_handling_configs}
        )
        self.logger.info("Trainable model setup complete.")

    def extract_trainable_params(self):
        #For now, lets put a dummy where it will just return all the network parameters.
        if self.network_architecture not in network_registry.keys():
            raise RuntimeError(f"Network architecture {self.network_architecture} not found in network registry, so can't be used for extracting trainable params!")
        elif self.network_architecture == 'nnInteractiveUNetFrozen':
            return [(name, param) for (name, param) in self.network.named_parameters() if param.requires_grad]
        elif self.network_architecture == 'nnInteractiveUNetTrainNorm':
            return [(name, param) for (name, param) in self.network.named_parameters() if param.requires_grad]
        elif self.network_architecture == 'nnInteractiveUNet':
            return self.network.parameters()

    def save_ckpt(self, is_best: bool, target_dir: str):
        #Saving the current training state to a checkpoint file in the tmp dir. 
        if self.network == None:
            raise RuntimeError("Network is None when trying to save checkpoint! Cannot save checkpoint without a model.")
        #Lets create the configs being stored #TODO: Needs updating with the optimisation state etc. 
        configs = {
            'current_epoch': self.stored_state['current_epoch'],
            'plans': self.stored_state['plans'],
            'prior_configs': { #We are going to just use this convention, so that it makes it easier to load prior models for adaptation. 
                #We are actually storing the current model configs here - > it becomes the prior config for the next adaptation episode.
                'model_architecture': self.network_architecture, 
                'input_encoding': self.planner_config['algorithm_config']['input_encoding'],
                'input_handling_configs': self.input_handling_configs,
                'network_configuration': self.network_configuration,
                'network_weights': self.network.state_dict(),
            }, 
            'train_dataloading_sampler_generator_state': self.train_dataloader.sampler.generator.get_state(),
            'train_dataloading_sampler_generator_preview': [int(next(iter(RandomSampler(self.train_dataloader.dataset, generator=self.train_dataloader.sampler.generator)))) for j in range(self.preview_num_samples)],
            'val_dataloading_sampler_generator_state': self.val_dataloader.sampler.generator.get_state(),
            'val_dataloading_sampler_generator_preview': [int(next(iter(RandomSampler(self.val_dataloader.dataset, generator=self.val_dataloader.sampler.generator)))) for j in range(self.preview_num_samples)],
            'global_iter_step': self.global_iter_step,
            'train_metric_history': self.train_metric_history,
            'val_metric_history': self.val_metric_history,
            'best_ema_metrics': self.best_ema_metrics,
            'best_metrics': self.best_metrics,
            'best_epoch': self.best_epoch,
            'optimiser_state_dict': self.optimiser.state_dict(),
            'current_lr_scheduler_name': self.current_lr_scheduler_name,
            'lr_scheduler_state_dict': self.current_lr_scheduler.state_dict(),
            'grad_scaler_state_dict': self.grad_scaler.state_dict()
        }
        #We flush the tensorboard writer to ensure all pending writes are done. 
        self.tensorboard_writer.flush() 

        #Save the checkpoint to last.pth, then write it to best.pth if it is best also. This gets triggered if we had a new best. we don't want to end up
        #with best.pth being in a later epoch than last.pth.
        save_checkpoint(
            temp_dir='temp',
            filename='last.pth',
            desired_dir=target_dir,
            configs=configs
        )
        if is_best: #If is_best then we also save best.pth
            self.logger.info(f"New best checkpoint at epoch {self.stored_state['current_epoch']}. Saving best.pth.")
            #We are going to be very careful and try to avoid corrupted checkpoints by first deleting the existing best.pth and writing a new one.
            os.remove(os.path.join(target_dir, 'best.pth')) #Removing existing best.pth first to avoid corruption. 
            #Now write a fresh version with our checkpointing function.
            save_checkpoint(
                temp_dir='temp',
                filename='best.pth',
                desired_dir=target_dir,
                configs=configs
            )

    #Actual trainer/val components setup functions.

    def setup_grad_scaler(self):
        self.grad_scaler = torch.amp.GradScaler(device='cuda')
        
    def setup_optimiser(self, train_plans):
        optimiser_conf = train_plans['optimisation_config']['optimiser']

        optimiser_factory = optimiser_registry.get(optimiser_conf['name'])
        optimiser_params = copy.deepcopy(optimiser_conf['params'])
        #Append the network params to the optimiser params.
        params_to_optimise = self.extract_trainable_params()
        optimiser_params['params']= params_to_optimise #Yes, different meaning of params here..., sorry.

        self.optimiser = optimiser_factory(
                optimiser_params
            )
        
        
        
    def setup_lr_schedulers(self, train_plans):
        lr_conf = train_plans['optimisation_config'].get('learning_rate_config')
        if len(lr_conf.get('scheduler_configs')) != 1:
            raise RuntimeError("Only single LR scheduler supported currently in trainer setup!") 
        if len(lr_conf.get('scheduler_configs')) == 0:
            raise RuntimeError("No LR scheduler configs found in training plans when setting up LR schedulers!")
        if len(lr_conf.get('scheduler_configs')) != len(lr_conf.get('scheduler_order_config')):
            raise RuntimeError("Number of LR schedulers and number of LR scheduler order configs do not match when setting up LR schedulers!")
        
        #We put this here to allow for future cases of using > 1 scheduler, but will require extra logic for the auto rerun when using multiple schedulers...? 
        self.lr_scheduler_collection = dict() 

        for name, scheduler_conf in lr_conf['scheduler_configs'].items():
            lr_factory = lr_scheduler_registry.get(name) 
            if lr_factory is None:
                raise RuntimeError(f"LR scheduler {name} not found in registry!") 
            self.lr_scheduler_collection[name] = lr_factory(
                {'optimizer':self.optimiser,
                **scheduler_conf['scheduler_params']}
            )
        self.lr_scheduler_order_conf = lr_conf['scheduler_order_config'] 
        #NOTE: 
        #Can use sequential LR scheduler class for future multi-scheduler implementations.

        if not self.resume:
            self.current_lr_scheduler_name = self.lr_scheduler_order_conf[0]['name']
        else:
            self.current_lr_scheduler_name = self.stored_state.get('current_lr_scheduler_name')
        
        if self.current_lr_scheduler_name is None:
            raise RuntimeError("No current LR scheduler name found!")
        self.current_lr_scheduler = self.lr_scheduler_collection[self.current_lr_scheduler_name] #NOTE: probably needs to change for re-run logic later.
        
    def update_lr(self, epoch, warmup_epoch=10, warm_up=False):
        raise NotImplementedError('LR update not implemented yet.')


    def setup_loss(self, train_plans):
        base_loss_conf = train_plans['loss_config'].get('base')
        if base_loss_conf is None:
            raise RuntimeError("No base loss config found in training plans when setting up loss!")
        if base_loss_conf.get('name') is None:
            raise RuntimeError(f"Base loss {base_loss_conf['name']} not found in loss registry!")
        if base_loss_conf.get('params') is None:
            raise RuntimeError(f"Base loss params for loss {base_loss_conf['name']} is None!")
        
        assert type(base_loss_conf.get('name')) == str 

        #Otherwise, we can set up the base loss function.
        base_loss_factory = loss_registry[base_loss_conf['name']]
        
        loss_kwargs = copy.deepcopy(base_loss_conf.get('params')) #We copy to avoid any in-place edits to the stored plans,
        #otherwise it will flag errors for changes to the plans mid-training despite it only being due to this reason...
        loss_kwargs.update({
                'configs_labels_dict': self.configs_labels_dict
            })
        self.base_loss = base_loss_factory(
            loss_kwargs
            )

        #Now lets set up the wrapper. 
        loss_wrapper_conf = train_plans['loss_config'].get('wrapper_config')
        if loss_wrapper_conf is None:
            raise RuntimeError("No loss wrapper config found in training plans when setting up loss!")
        if loss_wrapper_conf.get('used_outputs') is None:
            raise RuntimeError("No used outputs specified found in loss wrapper config when setting up loss!")
        if loss_wrapper_conf.get('merge_strategy') is None:
            raise RuntimeError("No merge strategy specified found in loss wrapper config when setting up loss!")
        if loss_wrapper_conf.get('early_exit_padding_strategy') is None:
            raise RuntimeError("No early exit padding strategy specified found in loss wrapper config when setting up loss!")
    
        self.loss_wrapper_conf = loss_wrapper_conf #Storing for use in loss calculation. 

    def setup_prompter(self, plans, plan_type: str):
        prompter_plans = plans['prompter']
        if plan_type == 'train_handlers':
            self.train_prompters = {mode: BuildHeuristic(
                sim_device=self.device,
                use_mem=prompter_plans['use_mem'],
                config_labels_dict=self.configs_labels_dict,
                heuristics=conf['prompter']['methods'],
                heuristic_params=conf['prompter']['build_params'],
                heuristic_mixtures=conf['prompter']['mixture_params'],
                heuristic_class_type=conf['prompter']['prompter_class_type']
            ) for mode, conf in prompter_plans['mode_configs'].items()}
            
            self.train_inner_loop_conf = {
                'max_interactions': prompter_plans['num_loop'],
            }

        elif plan_type == 'val_handlers':
            self.val_prompters = {mode: BuildHeuristic(
                sim_device=self.device,
                use_mem=prompter_plans['use_mem'],
                config_labels_dict=self.configs_labels_dict,
                heuristics=conf['prompter']['methods'],
                heuristic_params=conf['prompter']['build_params'],
                heuristic_mixtures=conf['prompter']['mixture_params'],
                heuristic_class_type=conf['prompter']['prompter_class_type']
            ) for mode, conf in prompter_plans['mode_configs'].items()}
            
            self.val_inner_loop_conf = {
                'max_interactions': prompter_plans['num_loop'],
            }
        else:
            raise RuntimeError(f"Unknown plan type {plan_type} when setting up prompter!")
        
    
    def setup_performance_tracking(self, val_plans):
        performance_plans = val_plans.get('performance_tracking_config')
        
        self.per_iter_train_metrics = dict()
        self.per_iter_train_metric_wrappers = dict()
        self.aggregate_train_metrics = dict()
        self.aggregate_train_metric_wrappers = dict()
        self.per_iter_val_metrics = dict()
        self.per_iter_val_metric_wrappers = dict()
        self.aggregate_val_metrics = dict()
        self.aggregate_val_metric_wrappers = dict()

        for metric_name, metric_conf in performance_plans.get('per_iter_train_metric_config').items():
            metric_factory = metric_registry.get(metric_name)
            if metric_factory is None:
                raise RuntimeError(f"Metric {metric_name} not found in registry!")
            self.per_iter_train_metrics[metric_name] = metric_factory(
                metric_conf['params']
            )
            self.per_iter_train_metric_wrappers[metric_name] = metric_conf['wrapper']
            if self.per_iter_train_metric_wrappers[metric_name] is None:
                raise RuntimeError(f"No wrapper specified for train metric {metric_name} when setting up performance tracking!")
             
        for metric_name, metric_conf in performance_plans.get('per_iter_val_metric_config').items():
            metric_factory = metric_registry.get(metric_name)
            if metric_factory is None:
                raise RuntimeError(f"Metric {metric_name} not found in registry!")
            self.per_iter_val_metrics[metric_name] = metric_factory(
                metric_conf['params']
            )

            self.per_iter_val_metric_wrappers[metric_name] = metric_conf['wrapper']
            if self.per_iter_val_metric_wrappers[metric_name] is None:
                raise RuntimeError(f"No wrapper specified for val metric {metric_name} when setting up performance tracking!")
        
        for metric_name, metric_conf in performance_plans.get('aggregate_train_metric_config').items():
            metric_factory = metric_registry.get(metric_name)
            if metric_factory is None:
                raise RuntimeError(f"Metric {metric_name} not found in registry!")
            self.aggregate_train_metrics[metric_name] = metric_factory(
                metric_conf['params']
            )
            self.aggregate_train_metric_wrappers[metric_name] = metric_conf['wrapper']
            if self.aggregate_train_metric_wrappers[metric_name] is None:
                raise RuntimeError(f"No wrapper specified for train metric {metric_name} when setting up performance tracking!")
        
        for metric_name, metric_conf in performance_plans.get('aggregate_val_metric_config').items():
            metric_factory = metric_registry.get(metric_name)
            if metric_factory is None:
                raise RuntimeError(f"Metric {metric_name} not found in registry!")
            self.aggregate_val_metrics[metric_name] = metric_factory(
                metric_conf['params']
            )
            self.aggregate_val_metric_wrappers[metric_name] = metric_conf['wrapper']
            if self.aggregate_val_metric_wrappers[metric_name] is None:
                raise RuntimeError(f"No wrapper specified for val metric {metric_name} when setting up performance tracking!")

        self.improvement_criterion_config = performance_plans.get('improvement_criterion_config')
        if self.improvement_criterion_config is None:
            raise RuntimeError("No improvement criterion found in validation plans when setting up performance tracking!")
        
    def __call__(
        self,
        logger,
        tensorboard_writer,
        configs_labels_dict: Dict[str, int],
        tmp_dir: str,
        split_name: str,
        dataloaders: dict[str, Any],
        resume: bool = False,
):
        
        self.resume = resume
        self.split_name = split_name
        self.logger = logger
        self.configs_labels_dict = configs_labels_dict
        self.tensorboard_writer = tensorboard_writer


        if self.resume:
            #We will look for the training state pkl file in the tmp dir to find the information needed
            #to resume training. 
            
            if not os.path.exists(os.path.join(tmp_dir, 'last.pth')):
                raise RuntimeError(f"Resume training but no training state last.pth file found in {tmp_dir}!")
            if not os.path.exists(os.path.join(tmp_dir, 'best.pth')):
                raise RuntimeError(f"Best.pth not found when epoch !=0! Absolutely cannot happen.")
            
            #Load the training state and resume training.
            self.stored_state = torch.load(
                os.path.join(tmp_dir, 'last.pth'), 
                weights_only=False
            )
            
            self.global_iter_step = self.stored_state.get("global_iter_step")
            if self.global_iter_step is None:
                raise RuntimeError("Tensorboard global step not found in stored state for resume!") 
            
            self.best_ema_metrics = self.stored_state.get("best_ema_metrics")
            if self.best_ema_metrics is None:
                raise RuntimeError("Best EMA metrics not found in stored state for resume!") 
            
            self.best_metrics = self.stored_state.get("best_metrics")
            if self.best_metrics is None:
                raise RuntimeError("Best metrics not found in stored state for resume!") 
            
            self.best_epoch = self.stored_state.get("best_epoch")
            if self.best_epoch is None:
                raise RuntimeError("Best epoch not found in stored state for resume!")

            self.train_metric_history = self.stored_state.get("train_metric_history") 
            if self.train_metric_history is None:
                raise RuntimeError("Train metric history not found in stored state for resume!") 
            self.val_metric_history = self.stored_state.get("val_metric_history") 
            if self.val_metric_history is None:
                raise RuntimeError("Validation metric history not found in stored state for resume!") 
            
            #Loading in the dataloaders to local variables.
            val_dataloader = dataloaders['val']
            train_dataloader = dataloaders['train']

            #Reloading the sampling state so that we can ensure that we don't imbalance the
            #training when resuming, especially for small datasets.
            #########################################
            sampler_train = getattr(train_dataloader, "sampler", None)
            sampler_val = getattr(val_dataloader, "sampler", None)
            if sampler_train is None:
                raise RuntimeError("Train DataLoader has no sampler; expected torch.utils.data.RandomSampler.")
            if sampler_val is None:
                raise RuntimeError("Validation DataLoader has no sampler; expected torch.utils.data.RandomSampler.")
            # enforce RandomSampler only
            if not isinstance(sampler_train, RandomSampler) or not isinstance(sampler_val, RandomSampler):
                raise RuntimeError(
                    f"Unsupported sampler type for resume: {type(sampler_train)} or {type(sampler_val)}. "
                    "Only torch.utils.data.RandomSampler is supported for deterministic resume on train, and same for val."
                )
            #We use a random sampler for both becaues for validation if we use an indexed sampler then it puts
            #more constraints on ensuring determinism. instead we want to lean on statistics, by resampling and
            #artificially increasing the set of patches to achieve this to some extent.


            #Now we will set the generator states. 
            train_dataloading_sampler_gen = self.stored_state.get("train_dataloading_sampler_generator_state") 
            train_saved_preview = self.stored_state.get("train_dataloading_sampler_generator_preview")
            if train_dataloading_sampler_gen is None or train_saved_preview is None:
                raise RuntimeError("Generator state not found in stored state") 
            else:
                #Lets do a check by previewing the permutation generated, and cross-referencing
                #it with a stored one. 
                preview_gen = torch.Generator(device="cpu")
                preview_gen.set_state(train_dataloading_sampler_gen)
                preview_first = [int(next(iter(RandomSampler(train_dataloader.dataset, generator=preview_gen)))) for j in range(self.preview_num_samples)]
                if preview_first != train_saved_preview:
                    raise RuntimeError("Dataloader generator state mismatch on resume! Cannot resume training.")

                #Now the check is done, lets recreate the dataloader.
                gen = torch.Generator(device="cpu")
                gen.set_state(train_dataloading_sampler_gen)
                sampler = RandomSampler(train_dataloader.dataset, generator=gen)
                train_dataloader = DataLoader(
                    train_dataloader.dataset,
                    batch_size=train_dataloader.batch_size,
                    sampler=sampler,
                    num_workers=train_dataloader.num_workers,
                    pin_memory=train_dataloader.pin_memory
                )

            #Now lets do the validation dataloader too.
            val_dataloading_sampler_gen = self.stored_state.get("val_dataloading_sampler_generator_state") 
            val_saved_preview = self.stored_state.get("val_dataloading_sampler_generator_preview")
            if val_dataloading_sampler_gen is None or val_saved_preview is None:
                raise RuntimeError("Validation generator state not found in stored state") 
            else:
                #Lets do a check by previewing the permutation generated, and cross-referencing
                #it with a stored one. 
                preview_gen = torch.Generator(device="cpu")
                preview_gen.set_state(val_dataloading_sampler_gen)
                preview_first = [int(next(iter(RandomSampler(val_dataloader.dataset, generator=preview_gen)))) for j in range(self.preview_num_samples)]
                if preview_first != val_saved_preview:
                    raise RuntimeError("Validation Dataloader generator state mismatch on resume! Cannot resume training.")

                #Now the check is done, lets recreate the dataloader.
                gen = torch.Generator(device="cpu")
                gen.set_state(val_dataloading_sampler_gen)
                sampler = RandomSampler(val_dataloader.dataset, generator=gen)
                val_dataloader = DataLoader(
                    val_dataloader.dataset,
                    batch_size=val_dataloader.batch_size,
                    sampler=sampler,
                    num_workers=val_dataloader.num_workers,
                    pin_memory=val_dataloader.pin_memory
                )

            #Now lets assign the dataloaders to class attributes.
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            
            #If there is a stored state on the dataloader RNG state, we will restore it here.

            if self.stored_state['plans'] != self.planner_config:
                raise RuntimeError(f"Adaptation plans do not match those in the stored training state! Cannot resume training.")    
            if self.stored_state['current_epoch'] <= 0:
                raise RuntimeError(f"Stored training state indicates epoch 0! Cannot resume training from epoch 0.")
            #################################################################################
            
            #Setting up training components.
            self.setup_training()
            #Setting up the validation components.
            self.setup_validation()

            #Lets dump the network weights from the state dict to free up VRAM. 
            self.stored_state['prior_configs']['network_weights'] = None
            torch.cuda.empty_cache()

            #####################################   Now we are ready to resume training.  #######################################################
            #Now execute the training. 
            self.stored_state['current_epoch'] += 1
            #Upper limit on the range is max_epochs + 1 because we start from 1 in our loop.
            if self.stored_state['current_epoch'] >= self.planner_config['epoch_config']['max_epochs'] + 1:
                self.logger.info(f"Training already completed in stored state. Current epoch: {self.stored_state['current_epoch']}, max epochs: {self.planner_config['epoch_config']['max_epochs']}. Nothing to do.")
                #In this case, somehow training is already complete but the remaining execution in the adaptation
                #executor was not, and so we just return here.  
                return 
            
            else:
                #We will use the logic of adding 1 to the epoch to be consistent. Epoch = 0 was the initial validation epoch. 
                for epoch_num in range(self.stored_state['current_epoch'], self.planner_config['epoch_config']['max_epochs'] + 1):
                    self.logger.info(f"Epoch {epoch_num} / {self.planner_config['epoch_config']['max_epochs']}")
                    #Storing epoch num in stored state for checkpoint saving.
                    self.stored_state['current_epoch'] = epoch_num 

                    self.train(epoch_num)
                    torch.cuda.empty_cache()
                    new_best = self.validate(epoch_num=epoch_num)
        #################################################################################
        
                    #Saving our checkpoints periodically so that we can resume training. 
                    if epoch_num % self.epoch_saving_period == 0 and not new_best:
                        self.save_ckpt(
                            is_best=False, #Bool which indicates we are saving this last checkpoint
                            target_dir=tmp_dir
                        )
                    if new_best:
                        #Saving the best checkpoint, it also saves the last checkpoint first to prevent
                        # mismatch/future checkpoints being ahead of last.  
                        self.save_ckpt(
                            is_best=True, #Bool which indicates we are saving a new best. 
                            target_dir=tmp_dir
                        )
                    


        else:
            #We are initialising training here. 
            self.stored_state = torch.load(
                os.path.join(tmp_dir, 'last.pth'), weights_only=False
            )
            if self.stored_state['current_epoch']!=0:
                raise RuntimeError(f"Stored training state indicates epoch !=0! Cannot initialise training from non-zero epoch.")
            
            #Loading in the dataloaders to local variables. 
            self.val_dataloader = dataloaders['val']
            self.train_dataloader = dataloaders['train'] 
            
            #First we start by running a validation epoch to get a baseline performance with the model
            #we are adapting on the current validation set (i.e. we do not want to misalign given that our validation set is going to be changing). 
            
            #So first, we must create a model instance and load the existing weights.
            self.setup_prior_model()
            #Setting up the validation components, so that we can run validation on the pre-existing model.
            self.setup_validation()
            self.validate(epoch_num=0)

            #Now we will drop the prior model from memory to free up VRAM.
            self.network = None
            torch.cuda.empty_cache()

            #Now we will initialise the "best" pth file using the starting checkpoint. 
            shutil.copyfile(
                os.path.join(tmp_dir, 'last.pth'),
                os.path.join(tmp_dir, 'best.pth')
            )
            #Setting up the training components.
            self.setup_training()
            
            #Lets dump the network weights from the state dict to free up VRAM. 
            self.stored_state['prior_configs']['network_weights'] = None
            torch.cuda.empty_cache()
            
            #####################################   Now we are ready to start training.  ####################################################### 
            for epoch_num in range(1, self.planner_config['epoch_config']['max_epochs'] + 1):
                self.logger.info(f"Starting epoch {epoch_num} / {self.planner_config['epoch_config']['max_epochs']}")
                #Storing epoch num in stored state for checkpoint saving.
                self.stored_state['current_epoch'] = epoch_num 

                self.train(epoch_num)
                torch.cuda.empty_cache()
                new_best = self.validate(epoch_num=epoch_num)
                #TODO: Implement logic for saving checkpoints here, etc.
                
                #Saving our checkpoints periodically so that we can resume training. 
                if epoch_num % self.epoch_saving_period == 0 and not new_best:
                    self.save_ckpt(
                        is_best=False, #Bool which indicates we are saving the last checkpoint ()
                        target_dir=tmp_dir
                    )
                if new_best:
                    #Saving the best checkpoint also. 
                    self.save_ckpt(
                        is_best=True, #Bool which indicates we are saving a new best. 
                        target_dir=tmp_dir
                    )


        self.network = None  #Freeing up memory.  
        torch.cuda.empty_cache() 


