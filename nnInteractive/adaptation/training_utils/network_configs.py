import os
from os.path import dirname as up 
import sys 
app_local_path = os.path.abspath(up(up(up(up(__file__)))))
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch 
from torch import nn
from nnInteractive.adaptation.training_utils.general_utils import make_factory 

#Lets just copy the stub structure and slightly modify it. 
class nnInteractiveUNet:
    def __init__(self, existing_kwargs:dict | None, current_kwargs:dict):
        if existing_kwargs is not None:
            raise NotImplementedError("nnInteractiveUNet class does not support adapting network config.")
        self.architecture_class_name = current_kwargs.get('network_arch_class_name')
        self.arch_init_kwargs = current_kwargs.get('network_arch_init_kwargs')
        self.arch_init_kwargs_req_import = current_kwargs.get('network_arch_init_kwargs_req_import')
        self.num_input_channels = current_kwargs.get('num_input_channels')
        self.num_output_channels = current_kwargs.get('num_output_channels')
        assert self.num_input_channels == 8, "1 image channel + 7, Added 7 channels for the interaction maps"
        assert self.num_output_channels == 2, "We assume binary segmentation for nnInteractive base model due to nnunet formatting"
        "still requires 2 output channels for binary segmentation."
        # self.num_output_channels = 2  # nnunet handles one class segmentation still as CE so we need 2 outputs.
        self.enable_deep_supervision = current_kwargs.get('enable_deep_supervision')        
        
        #Assert none of these are Nonetypes. 
        assert self.architecture_class_name is not None, "architecture_class_name cannot be None"
        assert self.arch_init_kwargs is not None, "arch_init_kwargs cannot be None"
        assert self.arch_init_kwargs_req_import is not None, "arch_init_kwargs_req_import cannot be None"
        assert self.num_input_channels is not None, "num_input_channels cannot be None"
        assert self.num_output_channels is not None, "num_output_channels cannot be None"
        assert self.enable_deep_supervision is not None, "enable_deep_supervision cannot be None" 
        
    def build_network_architecture(self, device: torch.device) -> torch.nn.Module:
        # architecture_class_name: str,
        # arch_init_kwargs: dict,
        # arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        # num_input_channels: int,
        # num_output_channels: int,
        # enable_deep_supervision: bool = True
        return nnUNetTrainer.build_network_architecture(
            self.architecture_class_name,
            self.arch_init_kwargs,
            self.arch_init_kwargs_req_import,
            self.num_input_channels, #Added 7 channels for interaction maps.
            self.num_output_channels,
            self.enable_deep_supervision
        ).to(device=device)

    def load_weights(self, device:torch.device, model, network_weights) -> torch.nn.Module:
        model.load_state_dict(
            network_weights
        )
        network_weights = None #Free up memory.
        torch.cuda.empty_cache()
        
        
        return model.to(device=device)

class nnInteractiveUNetFrozen(nnInteractiveUNet):
    def __init__(self, existing_kwargs:dict | None, current_kwargs:dict):
        super().__init__(existing_kwargs=None, current_kwargs=current_kwargs)
        #Existing kwargs are ignored for now because its the same network, just with frozen layers.
        self.existing_kwargs = existing_kwargs #We store it here, so that we can use it for building the network,
        #but we do not use it in the super init as this would break the upstream class initialisation. 
        self.current_kwargs = current_kwargs

    def build_network_architecture(self, device:torch.device) -> torch.nn.Module:
        model = super().build_network_architecture(device=device).to(device=device)

        #Normally we look at the discrepancy between existing and current kwargs to determine how to configure,
        #but this is just a dummy which freezes a part of the the network for debugging.
        num_named_params = len(list(model.named_parameters()))
        for idx, (name, param) in enumerate(model.named_parameters()):
            if idx != num_named_params - 1:
                param.requires_grad = False
        return model 
    
    def load_weights(self, device:torch.device, model, network_weights) -> torch.nn.Module:
        model.load_state_dict(
            network_weights
        )
        #NORMALLY we would need to add some additional logic to figure out where to load the weights but 
        #in this case its just the same architecture so we can directly load them.
        network_weights = None #Free up memory.
        torch.cuda.empty_cache()
        
        
        return model.to(device=device)

network_registry = {
    'nnInteractiveUNet': make_factory(nnInteractiveUNet),
    'nnInteractiveUNetFrozen': make_factory(nnInteractiveUNetFrozen)
}