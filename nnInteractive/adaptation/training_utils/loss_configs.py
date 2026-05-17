#Utilities for configuring base loss functions and wrappers on them for interactive seg. specific
# aspects for training.
from nnInteractive.adaptation.training_utils.general_utils import make_factory 
from monai.losses import DiceLoss 
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch
from monai.networks.utils import one_hot
from torch.nn import _reduction as _Reduction  
import torch.linalg as LA

def compute_tp_fp_fn(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce_axis: list[int],
    ord: int,
    soft_label: bool,
    decoupled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        input: the shape should be BNH[WD], where N is the number of classes.
        target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.
        reduce_axis: the axis to be reduced.
        ord: the order of the vector norm.
        soft_label: whether the target contains non-binary values (soft labels) or not.
            If True a soft label formulation of the loss will be used.
        decoupled: whether the input and the target should be decoupled when computing fp and fn.
            Only for the original implementation when soft_label is False.

    Adapted from:
        monai implementation 1.5.1
    """

    # the original implementation that is erroneous with soft labels
    if ord == 1 and not soft_label:
        tp = torch.sum(input * target, dim=reduce_axis)
        # the original implementation of Dice and Jaccard loss
        if decoupled:
            fp = torch.sum(input, dim=reduce_axis) - tp
            fn = torch.sum(target, dim=reduce_axis) - tp
        # the original implementation of Tversky loss
        else:
            fp = torch.sum(input * (1 - target), dim=reduce_axis)
            fn = torch.sum((1 - input) * target, dim=reduce_axis)
    # the new implementation that is correct with soft labels
    # and it is identical to the original implementation with hard labels
    else:
        pred_o = LA.vector_norm(input, ord=ord, dim=reduce_axis)
        ground_o = LA.vector_norm(target, ord=ord, dim=reduce_axis)
        difference = LA.vector_norm(input - target, ord=ord, dim=reduce_axis)

        if ord > 1:
            pred_o = torch.pow(pred_o, exponent=ord)
            ground_o = torch.pow(ground_o, exponent=ord)
            difference = torch.pow(difference, exponent=ord)

        tp = (pred_o + ground_o - difference) / 2
        fp = pred_o - tp
        fn = ground_o - tp

    return tp, fp, fn

def reduction_duplicate(size_average, reduce, reduction):
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    else:
        reduction = reduction
    return reduction 

class DiceCrossEntropyLoss:
    def __init__(self, **kwargs):
        self.name = 'DiceCrossEntropyLoss'
        self.semantic_id_dict = kwargs.get('semantic_id_dict')
        if self.semantic_id_dict == None:
            raise RuntimeError('semantic_id_dict parameter must be provided for DiceCrossEntropyLoss')
        
        
        #Next part is going to look weird, we have to make a workaround cuda determinism issues with the cross
        #entropy losses. We will set size_average, reduce, reduction to fixed values and ignore the user inputs for these.
        #This is because pytorch crossentropy losses have non-deterministic behaviour when using certain combinations of these parameters on cuda.
        #So we just fix them to size_average=False, reduce=False, reduction=None, which is equivalent to no reduction, and we will handle reduction ourselves if needed.
        

        #We then manually will handle the reduction outside of the loss functions if needed, 
        bce_params = kwargs.get('BinaryCrossEntropy')
        if bce_params == None:
            raise RuntimeError('BinaryCrossEntropy parameters must be provided for DiceCrossEntropyLoss')
        self.bce_loss = BCEWithLogitsLoss(
            weight=bce_params.get('weight'),
            size_average=False,#bce_params.get('size_average'),
            reduce=False,#bce_params.get('reduce'),
            reduction=None,#bce_params.get('reduction'),
            pos_weight=bce_params.get('pos_weight'),
        )

    
        ce_params = kwargs.get('CrossEntropy')
        if ce_params == None:
            raise RuntimeError('CrossEntropy parameters must be provided for DiceCrossEntropyLoss')
        self.ce_loss = CrossEntropyLoss(
            weight=ce_params.get('weight'),
            size_average=False,#ce_params.get('size_average'),
            ignore_index=ce_params.get('ignore_index'),
            reduce=False,#ce_params.get('reduce'),
            reduction=None,#ce_params.get('reduction'),
            label_smoothing=ce_params.get('label_smoothing'),
        )
        #Must do the reduction manually! 
        self.reduction_params_bce = {
            'size_average': bce_params.get('size_average'),
            'reduce': bce_params.get('reduce'),
            'reduction': bce_params.get('reduction'),
        }
        self.reduction_params_ce = {
            'size_average': ce_params.get('size_average'),
            'reduce': ce_params.get('reduce'),
            'reduction': ce_params.get('reduction'),
        } 
        
        assert ce_params.get('ignore_index') == -100, 'DiceCrossEntropyLoss requires ignore_index to be -100, so that we are not ignoring any class'
        
        dice_params = kwargs.get('Dice')
        self.weight = kwargs.get('weight')
        
        if dice_params == None:
            raise RuntimeError('Dice parameters must be provided for DiceCrossEntropyLoss')
        if self.weight == None:
            raise RuntimeError('weight parameter must be provided for DiceCrossEntropyLoss')

        assert dice_params.get('include_background') == False, 'DiceCrossEntropyLoss requires include_background to be False'
        'so that the background class is not dominating the loss'
        ' in order to enforce that the output has to already been one-hot encoded. '
        assert dice_params.get('to_onehot_y') == True, 'DiceLoss requires to_onehot_y to be True so that we can be '
        'pass through a single-channel target tensor.'
        
        if not dice_params['sigmoid'] and not dice_params['softmax']:
            raise RuntimeError('DiceLoss requires at least one of softmax or sigmoid to be True, '
                               'so that the output is transformed into probabilities')
        self.dice_loss = DiceLoss(
            include_background=dice_params.get('include_background'), 
            to_onehot_y=dice_params.get('to_onehot_y'), 
            sigmoid=dice_params.get('sigmoid'), 
            softmax=dice_params.get('softmax'),
            other_act=dice_params.get('other_act'), 
            squared_pred=dice_params.get('squared_pred'), 
            jaccard=dice_params.get('jaccard'), 
            reduction="none",  #We will handle reduction ourselves.
            smooth_nr=dice_params.get('smooth_nr'), 
            smooth_dr=dice_params.get('smooth_dr'), 
            batch=dice_params.get('batch'), 
            weight=dice_params.get('weight')
        )
        self.dice_to_onehot_y = dice_params.get('to_onehot_y')
        self.reduction_params_dice = {
            'reduction': dice_params.get('reduction')
        }

    def dice(self, output, target):
        loss = self.dice_loss(output, target.to(device=output.device))

        #Now applying a reduction.
        reduction = reduction_duplicate(
            None,
            None,
            self.reduction_params_dice['reduction']
        )
        if reduction == 'mean':
            loss = loss.mean(dim=[1,2,3,4])#This will compute the mean over the spatial dimensions.
        elif reduction == 'sum':
            raise Exception('Sum reduction is not stable unless averaged on spatial dimensions first.')
        torch.cuda.empty_cache()
        return loss
    
    def bce(self, output, target):
        assert output.shape == target.shape, "For BCE loss, output and target must have the same shape"
        loss = self.bce_loss(output, target)
        #Now applying a reduction.
        reduction = reduction_duplicate(
            self.reduction_params_bce['size_average'],
            self.reduction_params_bce['reduce'],
            self.reduction_params_bce['reduction']
        )
        if reduction == 'mean':
            loss = loss.mean(dim=list(range(1, loss.ndim))) #This will compute the mean over the spatial dimensions.
        elif reduction == 'sum':
            raise Exception('Sum reduction is not stable unless averaged on spatial dimensions first.')
        return loss
    
    def ce(self, output, target):
        assert target.shape[1] == 1, "Target is expected to have a single channel/not be one-hot encoded"  
        target = target.squeeze(dim=1)
        loss = self.ce_loss(output, target.to(dtype=torch.long))
        #Now applying a reduction.
        reduction = reduction_duplicate(
            self.reduction_params_ce['size_average'],
            self.reduction_params_ce['reduce'],
            self.reduction_params_ce['reduction']
        )
        if reduction == 'mean':
            loss = loss.mean(dim=list(range(1, loss.ndim))) #This will compute the mean over the spatial dimensions.
        elif reduction == 'sum':
            raise Exception('Sum reduction is not stable unless averaged on spatial dimensions first.')
        return loss
    def __call__(self, output, target):
        #We just check that the output and target have the same shape.
        assert output.shape[2:] == target.shape[2:], "Output and target must have the same spatial shape"
        #we also assume a BCHWD shape for calculating losses. 
        assert output.ndim == 5
        assert output.shape[1] == len(self.semantic_id_dict) #We need the output logits to have 
        #the same number of channels as the number of labels we are predicting.
        
        if not target.shape[1] == 1:
            raise RuntimeError('Target tensor must have a single channel for DiceCrossEntropyLoss')

        if self.dice_to_onehot_y: #This is for the dice loss. 
            if output.shape[1] == 1:
                raise RuntimeError('Output has only one channel, but target needs to be one-hot encoded. Cannot proceed.') 
  
        assert output.shape[0] == target.shape[0]  #Batch size must be the same.

        dice_loss = self.dice(output, target.to(device=output.device))
        if output.shape[1] == 1: #Then we can use the default bce loss.
            bce_loss = self.bce(output.squeeze(dim=1), target.squeeze(dim=1).to(device=output.device))
            total_loss = self.weight['Dice'] * dice_loss + self.weight['BCE'] * bce_loss
        elif output.shape[1] > 1: #Then we just use the CE, even if its for binary case.
            ce_loss = self.ce(output, target.to(device=output.device))
            total_loss = self.weight['Dice'] * dice_loss + self.weight['CE'] * ce_loss
        
        torch.cuda.empty_cache()
        return {idx: total_loss[idx] for idx in range(total_loss.shape[0])}



loss_registry = {
    # 'CELoss': make_factory(CrossEntropyLoss),
    'DiceCELoss': make_factory(DiceCrossEntropyLoss),
    # 'FocalLoss': make_factory(FocalLoss),
    # 'DiceLoss': make_factory(DiceLoss),
}