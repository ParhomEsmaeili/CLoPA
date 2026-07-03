from clopa.adaptation.training_utils.general_utils import make_factory
from clopa.adaptation.training_utils.loss_configs import compute_tp_fp_fn
from scipy import integrate
from monai.networks.utils import one_hot
import torch 
from monai.data import MetaTensor
from typing import Union
import re 

def sort_infer_calls(infer_call_names):
    '''
    This function sorts the inference call names, and outputs them in a tuple format such that they are immutable.
    '''
    if not isinstance(infer_call_names, set):
        raise TypeError('The input infer call names must be a set data structure, please convert it to a set before passing it through!')
    if len(infer_call_names) < 1:
        raise Exception(f'At least one infer mode call subdict is required for metrics to be saved!')
    
    #We do not assume that the inference call names (or dict they were taken from) were ordered correctly, 
    # even if it is unlikely to be incorrectly ordered.

    infer_call_names_order = []
    #Check if there is an initialisation: if so, place that first. 
    init_modes  = {'init'}

    if init_modes & infer_call_names:
        #If the set is not empty
        if len(init_modes & infer_call_names) > 1:
            raise Exception('Cannot have two conflicting initialisation modes')
        else:
            infer_call_names_order.extend(init_modes & infer_call_names)
        
    #We already implemented a check to ensure that the infer call names are not empty! 

    #Therefore, we just sort and append according to the iteration num of the edit iter. First finding asymmetric
    #set diff.

    edit_names_list = list(infer_call_names.difference(init_modes))
    #Sorting this list.
    edit_names_list.sort(key=lambda test_str : list(map(int, re.findall(r'\d+', test_str))))
    
    #Extending the infer call ordered list. 
    #
    infer_call_names_order.extend(edit_names_list) 
    
    #Returning it as a tuple so that it is immutable.

    return tuple(infer_call_names_order)



class DiceMetric:
    def __init__(self, **kwargs):
        self.include_background = kwargs.get('include_background')
        self.classwise_agg = kwargs.get('classwise_agg') #This parameter controls how we handle the metric
        # on a class basis (i.e, do we aggregate and then calc the metric, or calc per class then average)
        
        self.batchwise_agg = kwargs.get('batchwise_agg') #This paramater controls whether we calculate the metric
        #on a per-batch basis or per-sample basis prior to reduction. I.e., calculate the dice and then take the
        #mean over the batch, or just calculte the dice on a full batch at once.
        
        self.ignore_empty = kwargs.get('ignore_empty')
    
        assert self.classwise_agg in [True, False], "classwise must be a boolean"
        assert self.include_background in [True, False], "include_background must be a boolean"
        assert self.batchwise_agg in [True, False], "batchwise must be a boolean"
        assert self.ignore_empty in [True, False], "ignore_empty must be a boolean"
         
    def __call__(self, predictions, targets, semantic_id_dict):
        # predictions = B X 1 X H X W X D
        # targets = B X 1 X H X W X D
        assert predictions.ndim == targets.ndim, "Predictions and targets must have the same number of dimensions"
        assert predictions.shape == targets.shape
        assert targets.shape[1] == 1, "pred/target must have a channel dimension of size 1, i.e., not be already one-hot encoded"
        
      
        #One-hot encode the target, so that it has the same number of channels as the predictions.
        num_classes = len(semantic_id_dict)
        predictions = one_hot(predictions, num_classes=num_classes)
        targets = one_hot(targets, num_classes=num_classes)

        assert predictions.shape == targets.shape, "After one-hot encoding, predictions and targets must have the same shape"
        
        if self.include_background == False:
            #Remove the background class from the predictions and targets.
            predictions = predictions[:,1:,...]
            targets = targets[:,1:,...]
        
        if self.classwise_agg and self.batchwise_agg:
            #In this case, we are aggregating over the batch and the classes before calculating the dice.
            tp, fp, fn = compute_tp_fp_fn(
                input=predictions,
                target=targets,
                reduce_axis=[0,1] + list(range(2, predictions.ndim)), #Reduce over all axes
                ord=1,
                soft_label=False,
                decoupled=True
            )
        
        elif not self.classwise_agg and self.batchwise_agg:
            #In this case, we are aggregating over the batch, but not the classes, before calculating the dice.
            raise NotImplementedError("DiceMetric currently does not support non-classwise but batchwise aggregated calculation.")
        
        elif self.classwise_agg and not self.batchwise_agg:
            #In this case, we are aggregating over the classes, but not the batch, before calculating the dice.
            tp, fp, fn = compute_tp_fp_fn(
                input=predictions,
                target=targets,
                reduce_axis=[1] + list(range(2, predictions.ndim)), #Reduce over the channel axis only + spatial axes.
                ord=1,
                soft_label=False,
                decoupled=True
            )
            
        elif not self.classwise_agg and not self.batchwise_agg:
            #In this case, we are not aggregating over classes or batch before calculating the dice.
            raise NotImplementedError("DiceMetric currently does not support non-classwise and non-batchwise aggregated calculation.")
        

        num = 2 * tp
        denom = 2 * (tp + fp + fn)

        if self.ignore_empty:
            #In this case, then we will ignore the nans produced by empty fg gt samples..
            raise NotImplementedError("DiceMetric currently does not support ignore_empty=True functionality.")
            #We choose not to support this because ultimately its not as robust. 
            #Better to just filter out empty samples a-priori, and to just consider whether the overlap was done
            #correctly AND consistently (in the absence of fg gt the correct thing is to be fully background!)
        else:
            dice_scores = num / denom  # shape: (B,)  
            dice_scores = dice_scores.to(torch.device('cpu'))
            
            del tp, fp, fn, denom, predictions, targets
            assert dice_scores.ndim == 1, "Dice scores must be of shape (B,)"
            #If any is nan, then we must check whether the numerator for that batch was also 0.
            if torch.isnan(dice_scores).any():
                nan_mask = torch.isnan(dice_scores)
                if nan_mask.ndim == 0:
                    if num == torch.zeros(dice_scores.shape[0]):
                        del num 
                        torch.cuda.empty_cache()
                        return torch.ones(dice_scores.shape[0], device='cpu') #If both num and denom are 0, then we set dice to 1.
                    else:
                        #Then all were not zero despite no nan flagged, raise zero.
                        del num 
                        torch.cuda.empty_cache()
                        return torch.zeros(dice_scores.shape[0], device='cpu') #If num is not zero, but denom is zero, then we set dice to 0.
                elif nan_mask.ndim == 1:
                    for i in range(dice_scores.shape[0]):
                        if nan_mask[i]:
                            if num[i] == 0:
                                dice_scores[i] = 1.0 #If both num and denom are 0, then we set dice to 1.
                            else:
                                dice_scores[i] = 0.0 #If num is not zero, but denom is zero, then we set dice to 0.
                    del num 
                    torch.cuda.empty_cache()
                    return dice_scores
                else:
                    raise NotImplementedError("DiceMetric currently does not support nan handling for dice scores with ndim > 1.")
        
            else:
                del num
                torch.cuda.empty_cache()
                return dice_scores
            
class DiceAUCMetric:
    def __init__(self, **kwargs):
        self.batchwise_reduce = kwargs.get('batchwise_reduce')
        #whether we will reduce 
        self.iterations_used = kwargs.get('iteration') #Can be 'all' 
        assert self.batchwise_reduce in [True, False], "batchwise must be a boolean"
        assert self.iterations_used in ['all'], "iterations_used must be 'all' for now."
        
    def __call__(
            self, 
            per_iter_metrics: dict,
            wrapper_conf: dict, 
            batchsize: int):
        #Per iter metrics is a dict consisting of the per-iteration dice scores. the batch level dice scores need
        #not necessarily be the same shape, as some batch samples may converge earlier than others.
        #We will therefore calculate the AUC for each batch sample individually, and then average over the batch.

        #Lets filter the relevant iterations.
        if self.iterations_used == 'all':
            per_iter_metrics = {k: v for k, v in per_iter_metrics.items() if k != 'final'} #We filter out the "final" 
            #as that is just a metric which is calculated on the final state of each sample, and so will be redundant/
            # double counting for AUC calculation.    
        else:
            raise NotImplementedError("DiceAUCMetric currently only supports 'all' iterations_used.")
        
        #Lets extract the per-iteration dice scores, we will construct the order of the iteration names.

        infer_call_names_order = sort_infer_calls(set(per_iter_metrics.keys()))

        per_iter_metrics = {
            b_idx: [per_iter_metrics[iter_name]['DiceMetric'][b_idx] if ('DiceMetric' in per_iter_metrics[iter_name] and b_idx in per_iter_metrics[iter_name]['DiceMetric']) else torch.nan for iter_name in infer_call_names_order]  #List of tensors, we will assume
        #that the order is consistent with the interaction order
        for b_idx in range(batchsize)
        }

        if per_iter_metrics == {}:
            raise ValueError("No per-iteration DiceMetric found in per_iter_metrics for DiceAUCMetric calculation."
            "There should be at least one per-iteration metric available, otherwise it meant no sample was eligible OR"
            "that the per-iteration DiceMetric was not being calculated/stored")
        if any([len(scores) == 0 for scores in per_iter_metrics.values()]):
            raise ValueError("At least one batch sample has no per-iteration DiceMetric scores for DiceAUCMetric calculation."
            "This meant that no iterations were eligible for that sample, or that the per-iteration DiceMetric was not being calculated/stored")
        if any(torch.all(torch.tensor([torch.isnan(torch.tensor(i)) for i in scores])) for scores in per_iter_metrics.values()):
            raise ValueError("At least one batch sample has all NaN per-iteration DiceMetric scores for DiceAUCMetric calculation."
            "This meant that no iterations were eligible for that sample, or that the per-iteration DiceMetric was not being calculated/stored")
        
        #We will assert that the batch size is non-increasing. 
        if any(
            torch.isnan(torch.tensor(per_iter_metrics[b_idx][i])) and not torch.isnan(torch.tensor(per_iter_metrics[b_idx][i+1]))
            for i in range(len(infer_call_names_order)-1)
            for b_idx in range(batchsize)
        ):
            raise ValueError("Batch size must be non-increasing over iterations for DiceAUCMetric calculation.")
        
        #Now lets group the per-iteration dice scores by batch sample. We will filter out the nans for each sample.
        aucs = dict()
        for b_idx, dice_scores in per_iter_metrics.items():
            #Filtering out nans
            filtered_scores = [score.cpu() for score in dice_scores if not torch.isnan(torch.tensor(score))]
            if len(filtered_scores) == 1:
                #If there is only one score, then we cannot calculate the AUC.
                aucs[b_idx] = torch.tensor(filtered_scores[0])
            elif len(filtered_scores) > 1:
                #Calculating the normalised AUC using trapezoidal rule. It is normalised by len - 1 because, that gives us the actual number of intervals. The aggregation happens on these intervals.
                normalised_auc = integrate.trapezoid(filtered_scores)/(len(filtered_scores) - 1)  
                aucs[b_idx] = torch.tensor(normalised_auc)
            else:
                raise ValueError("No valid DiceMetric scores found for batch sample {b_idx} for DiceAUCMetric calculation."
                "This meant that no iterations were eligible for that sample, or that the per-iteration DiceMetric was not being calculated/stored" \
                "this should have already been caught by previous checks.")
        #Now we will have to reduce over the batch.
        if self.batchwise_reduce:
            output = torch.stack(list(aucs.values()), dim=0).mean()  # shape: (B,)
            assert output.device == torch.device('cpu'), "DiceAUCMetric output must be on CPU."
        else:
            output = aucs  # shape: (B,)
            assert all([v.device == torch.device('cpu') for v in output.values()]), "DiceAUCMetric output must be on CPU."
        return output

metric_registry = {
    'DiceMetric': make_factory(DiceMetric),
    'DiceAUCMetric': make_factory(DiceAUCMetric)
}