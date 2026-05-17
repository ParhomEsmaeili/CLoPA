import os
from os.path import dirname as up 
import sys 
app_local_path = os.path.abspath(up(up(up(up(__file__)))))
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch 
from torch import nn
from functools import partial
from torch.utils.checkpoint import checkpoint 
from nnInteractive.adaptation.training_utils.general_utils import make_factory 

class CheckpointedModule(nn.Module):
    def __init__(self, module, use_reentrant: bool = False):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant

    def forward(self, *args):
        # only positional tensor args; avoid non-tensor kwargs here
        print(f"[CHECKPOINT] Running {self.module.__class__.__name__}") 
        return checkpoint(self.module, *args, use_reentrant=self.use_reentrant)


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

class nnInteractiveUNetFrozenDebugging(nnInteractiveUNet):
    '''
    This is a dummy debugging class which freezes all but the last named parameter of the network. 
    This is just for testing the training pipeline, and serves as a template for more complex freezing logic.
    '''
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


class nnInteractiveUNetInstanceNorm(nnInteractiveUNet):
    '''
    nnInteractiveUNetInstanceNorm only adjusts the affine parameters on the instance norm layers.
    '''
    def __init__(self, existing_kwargs:dict | None, current_kwargs:dict):
        super().__init__(existing_kwargs=None, current_kwargs=current_kwargs)
        #Existing kwargs are ignored for now because its the same network, just with batchnorm layers.
        self.existing_kwargs = existing_kwargs #We store it here, so that we can use it for building the network,
        #but we do not use it in the super init as this would break the upstream class initialisation. 
        self.current_kwargs = current_kwargs

    def apply_forward_checkpoint(self, model):
        # network_structure = [['encoder','decoder'],[[['stages', 'stages'], ['0','1','2','3','4','5']], [['stages']]]]
        # network_structure = [['encoder','decoder'],[[['stages', 'stages', 'stages', 'stages', 'stages'], ['0','1','2','3','4','5']], [['stages']]]]
        network_structure = [['decoder'],[[['stages']]]]
        
        if not all(i[0][0] == 'stages' for i in network_structure[1]):
            raise NotImplementedError("Only implemented for 'stages' network structure.") 
            
        #Lets only checkpoint the decoder for now as the skip connections
        #will make it more memory intensive.

        #Lets assign a set of "levels" which we will permit checkpointing within. Even then, we will
        #not probably checkpoint the entire block, just the most memory intensive activations. 
        
        #Now at the intra-level we can assign blocks which we will checkpoint.

        #Lets assign the blocks within a "level" which can be checkpointed also.
        # encoder_blocks = ['blocks', ['conv1', 'conv2', 'conv3', 'skip']] #nnu-net uses the same overall naming convention within
        encoder_blocks = ['blocks']

        decoder_blocks = ['convs', ['conv']] #nnu-net uses the same overall naming convention within
        #a block, so we can use this to iterate over the blocks and checkpoint specific layers.
        if encoder_blocks[0] != 'blocks' or decoder_blocks[0] != 'convs':
            raise NotImplementedError("Only implemented for 'blocks' and 'convs' intra-level structure.")
        

        #it is not consistent between the encoder and decoder though, so we need to be careful.
        
        for name, module in model.named_modules():
            parts = name.split('.') #Lets split by dot to get the module hierarchy.
            #We will filter valid modules by order of hierarchy.
            #we create a list which tracks at each level if we have matched a module.
            if len(parts) < len(network_structure):
                continue #Too short to match.
            #Now lets iterate over the encoder/decoder structure.
            for idx, struct in enumerate(network_structure[0]):
                if struct == 'decoder':
                    dummy_network_structures = [[struct] + network_structure[1][idx][0]]
                elif struct == 'encoder':
                    # This was for looking within each block within the stage.
                    dummy_network_structures = [[struct] + [i, j] for i, j in zip(network_structure[1][idx][0], network_structure[1][idx][1])]
                   
                else:
                    raise ValueError("Unknown network structure for checkpointing.")
                
                for dummy_network_structure in dummy_network_structures:
                    if len(parts) < len(dummy_network_structure):
                        continue #Too short to match.
                    structure_checker = [parts[i] == dummy_network_structure[i] for i in range(len(dummy_network_structure))]
                    if not all(structure_checker):
                        continue
                    
                    #If structure matches now we look at intra level
                    if dummy_network_structure[0] == 'decoder':
                        look_here = decoder_blocks
                    elif dummy_network_structure[0] == 'encoder':
                        look_here = encoder_blocks
                    else:
                        raise ValueError("Unknown network structure for checkpointing.")
                    blockparts = parts[len(dummy_network_structure):]

                    if dummy_network_structure[0] == 'decoder':
                        if len(blockparts) != 2 * len(look_here): #Stages have indexing, and so do the blocks within
                            #each level.
                            continue #Not the right level to match, either too coarse or too granular. 
                    elif dummy_network_structure[0] == 'encoder':
                        #Here we actually are going to checkpoint entire blocks so we need to 
                        #adjust our granularity.
                        if len(blockparts) != len(look_here) + 1: #Stages have indexing, and so do the blocks within
                            #each level. We are going to be looking at entire blocks here, nothing more granular.
                            continue #Not the right level to match, either too coarse or too granular.

                    # #Lets look past indexing of the stages.
                    # if blockparts[1] != look_here[0]:
                    #     continue
                    # #Within the block, we also have indexing of conv blocks.
                    # if blockparts[2] != look_here[1]:
                    #     continue
                    #We wrap both checks into one for loop.
                    if struct == 'encoder':
                        #This old tracker was for checkpointing within each block within a stage.
                        # intra_tracker = [blockparts[i * 2] in look_here[i] for i in range(len(look_here))]
                        intra_tracker = [blockparts[0] == look_here[0]] #We checkpoint entire blocks here. 
                    elif struct == 'decoder':
                        intra_tracker = [blockparts[i * 2 + 1] in look_here[i] for i in range(len(look_here))]
                    if not all(intra_tracker):
                        continue
                    else:
                        #We have a match, we will checkpoint this module's forward pass.
                        try:
                            block = model.get_submodule(name)
                            # # Replace the block with a checkpointed version
                            # parent_path, block_name = name.rsplit('.', 1)
                            # parent = model.get_submodule(parent_path) if parent_path else model
                            # setattr(parent, block_name, CheckpointedModule(block, use_reentrant=False))
                            # print(f"✓ Checkpointed {name}")
                            # Capture ORIGINAL forward before ANY replacement
                            original_forward = block.__class__.forward.__get__(block, block.__class__)
                            
                            # Now replace with checkpointed version
                            def make_checkpointed(orig, block_name):
                                def forward_fn(*args, **kwargs):
                                    print(f"[CHECKPOINT] Running checkpointed {block_name}")
                                    return checkpoint(orig, *args, use_reentrant=False, **kwargs)
                                return forward_fn
                            
                            block.forward = make_checkpointed(original_forward, name)
                        except Exception as e:
                            print(f"✗ Failed to checkpoint {name}: {e}")

            
        return model
    
    def apply_forward_checkpoint_test_orig(self, model):
        orig_forward = model.forward
        def wrapped_forward(*args):
            return checkpoint(partial(orig_forward), *args, use_reentrant=False)  # kwargs avoided; use partial if needed
        model.forward = wrapped_forward
        return model 
    def build_network_architecture(self, device:torch.device) -> torch.nn.Module:
        model = super().build_network_architecture(device=device).to(device=device)

        #Normally we look at the discrepancy between existing and current kwargs to determine how to configure,
        #but this is just a dummy which adds batchnorm layers for debugging.
        
        # First, freeze ALL parameters in the entire model
        for param in model.parameters():
            param.requires_grad = False
        
        # Then, unfreeze only the affine parameters (weight and bias) of InstanceNorm3d layers
        for module in model.modules():
            if isinstance(module, nn.InstanceNorm3d):
                if module.affine:
                    module.weight.requires_grad = True
                    module.bias.requires_grad = True
                else:
                    raise ValueError("InstanceNorm3d layer does not have affine parameters to train.") 
        
        #Lets try and checkpoint some operations so that we don't run out of VRAM....
        #slower but at least it won't BREAK. This will need to overwrite the forward function of
        #the prior model though, which is annoying. 
        # model = self.apply_forward_checkpoint(model)
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

class nnInteractiveUNetConv(nnInteractiveUNet):
    '''
    nnInteractiveUNetConv adjusts only the the encoder stem conv blocks, and the decoder layer conv blocks.
    '''
    def __init__(self, existing_kwargs:dict | None, current_kwargs:dict):
        super().__init__(existing_kwargs=None, current_kwargs=current_kwargs)
        #Existing kwargs are ignored for now because its the same network, just with batchnorm layers.
        self.existing_kwargs = existing_kwargs #We store it here, so that we can use it for building the network,
        #but we do not use it in the super init as this would break the upstream class initialisation. 
        self.current_kwargs = current_kwargs

    def apply_forward_checkpoint(self, model):
        # network_structure = [['encoder','decoder'],[[['stages', 'stages'], ['0','1','2','3','4','5']], [['stages']]]]
        # network_structure = [['encoder','decoder'],[[['stages', 'stages', 'stages', 'stages', 'stages'], ['0','1','2','3','4','5']], [['stages']]]]
        network_structure = [['decoder'],[[['stages']]]]
        
        if not all(i[0][0] == 'stages' for i in network_structure[1]):
            raise NotImplementedError("Only implemented for 'stages' network structure.") 
            
        #Lets only checkpoint the decoder for now as the skip connections
        #will make it more memory intensive.

        #Lets assign a set of "levels" which we will permit checkpointing within. Even then, we will
        #not probably checkpoint the entire block, just the most memory intensive activations. 
        
        #Now at the intra-level we can assign blocks which we will checkpoint.

        #Lets assign the blocks within a "level" which can be checkpointed also.
        # encoder_blocks = ['blocks', ['conv1', 'conv2', 'conv3', 'skip']] #nnu-net uses the same overall naming convention within
        encoder_blocks = ['blocks']

        decoder_blocks = ['convs', ['conv']] #nnu-net uses the same overall naming convention within
        #a block, so we can use this to iterate over the blocks and checkpoint specific layers.
        if encoder_blocks[0] != 'blocks' or decoder_blocks[0] != 'convs':
            raise NotImplementedError("Only implemented for 'blocks' and 'convs' intra-level structure.")
        

        #it is not consistent between the encoder and decoder though, so we need to be careful.
        
        for name, module in model.named_modules():
            parts = name.split('.') #Lets split by dot to get the module hierarchy.
            #We will filter valid modules by order of hierarchy.
            #we create a list which tracks at each level if we have matched a module.
            if len(parts) < len(network_structure):
                continue #Too short to match.
            #Now lets iterate over the encoder/decoder structure.
            for idx, struct in enumerate(network_structure[0]):
                if struct == 'decoder':
                    dummy_network_structures = [[struct] + network_structure[1][idx][0]]
                elif struct == 'encoder':
                    # This was for looking within each block within the stage.
                    dummy_network_structures = [[struct] + [i, j] for i, j in zip(network_structure[1][idx][0], network_structure[1][idx][1])]
                   
                else:
                    raise ValueError("Unknown network structure for checkpointing.")
                
                for dummy_network_structure in dummy_network_structures:
                    if len(parts) < len(dummy_network_structure):
                        continue #Too short to match.
                    structure_checker = [parts[i] == dummy_network_structure[i] for i in range(len(dummy_network_structure))]
                    if not all(structure_checker):
                        continue
                    
                    #If structure matches now we look at intra level
                    if dummy_network_structure[0] == 'decoder':
                        look_here = decoder_blocks
                    elif dummy_network_structure[0] == 'encoder':
                        look_here = encoder_blocks
                    else:
                        raise ValueError("Unknown network structure for checkpointing.")
                    blockparts = parts[len(dummy_network_structure):]

                    if dummy_network_structure[0] == 'decoder':
                        if len(blockparts) != 2 * len(look_here): #Stages have indexing, and so do the blocks within
                            #each level.
                            continue #Not the right level to match, either too coarse or too granular. 
                    elif dummy_network_structure[0] == 'encoder':
                        #Here we actually are going to checkpoint entire blocks so we need to 
                        #adjust our granularity.
                        if len(blockparts) != len(look_here) + 1: #Stages have indexing, and so do the blocks within
                            #each level. We are going to be looking at entire blocks here, nothing more granular.
                            continue #Not the right level to match, either too coarse or too granular.

                    # #Lets look past indexing of the stages.
                    # if blockparts[1] != look_here[0]:
                    #     continue
                    # #Within the block, we also have indexing of conv blocks.
                    # if blockparts[2] != look_here[1]:
                    #     continue
                    #We wrap both checks into one for loop.
                    if struct == 'encoder':
                        #This old tracker was for checkpointing within each block within a stage.
                        # intra_tracker = [blockparts[i * 2] in look_here[i] for i in range(len(look_here))]
                        intra_tracker = [blockparts[0] == look_here[0]] #We checkpoint entire blocks here. 
                    elif struct == 'decoder':
                        intra_tracker = [blockparts[i * 2 + 1] in look_here[i] for i in range(len(look_here))]
                    if not all(intra_tracker):
                        continue
                    else:
                        #We have a match, we will checkpoint this module's forward pass.
                        try:
                            block = model.get_submodule(name)
                            # # Replace the block with a checkpointed version
                            # parent_path, block_name = name.rsplit('.', 1)
                            # parent = model.get_submodule(parent_path) if parent_path else model
                            # setattr(parent, block_name, CheckpointedModule(block, use_reentrant=False))
                            # print(f"✓ Checkpointed {name}")
                            # Capture ORIGINAL forward before ANY replacement
                            original_forward = block.__class__.forward.__get__(block, block.__class__)
                            
                            # Now replace with checkpointed version
                            def make_checkpointed(orig, block_name):
                                def forward_fn(*args, **kwargs):
                                    print(f"[CHECKPOINT] Running checkpointed {block_name}")
                                    return checkpoint(orig, *args, use_reentrant=False, **kwargs)
                                return forward_fn
                            
                            block.forward = make_checkpointed(original_forward, name)
                        except Exception as e:
                            print(f"✗ Failed to checkpoint {name}: {e}")

            
        return model
    
    def apply_forward_checkpoint_test_orig(self, model):
        orig_forward = model.forward
        def wrapped_forward(*args):
            return checkpoint(partial(orig_forward), *args, use_reentrant=False)  # kwargs avoided; use partial if needed
        model.forward = wrapped_forward
        return model 
    def build_network_architecture(self, device:torch.device) -> torch.nn.Module:
        model = super().build_network_architecture(device=device).to(device=device)

        #Normally we look at the discrepancy between existing and current kwargs to determine how to configure,
        #but this is just a dummy which adds batchnorm layers for debugging.
        
        # First, freeze ALL parameters in the entire model
        for param in model.parameters():
            param.requires_grad = False
        
        # Then, unfreeze only the affine parameters (weight and bias) of InstanceNorm3d layers
        for name, module in model.named_modules():
            #Now lets unfreeze the conv blocks too. We will restrict this to the stem and decoder seg layer conv blocks for now, as these are the most relevant for adaptation and also to limit the number of trainable parameters.
            if isinstance(module, nn.Conv3d) and ('stem' in name or 'seg_layers' in name): 
                module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True

        #Lets try and checkpoint some operations so that we don't run out of VRAM....
        #slower but at least it won't BREAK. This will need to overwrite the forward function of
        #the prior model though, which is annoying. 
        # model = self.apply_forward_checkpoint(model)
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
    
class nnInteractiveUNetConvNorm(nnInteractiveUNet):
    '''
    nnInteractiveUNetConvNorm adjusts the affine parameters on the instance norm layers, the encoder stem conv blocks, and the .
    '''
    def __init__(self, existing_kwargs:dict | None, current_kwargs:dict):
        super().__init__(existing_kwargs=None, current_kwargs=current_kwargs)
        #Existing kwargs are ignored for now because its the same network, just with batchnorm layers.
        self.existing_kwargs = existing_kwargs #We store it here, so that we can use it for building the network,
        #but we do not use it in the super init as this would break the upstream class initialisation. 
        self.current_kwargs = current_kwargs

    def apply_forward_checkpoint(self, model):
        # network_structure = [['encoder','decoder'],[[['stages', 'stages'], ['0','1','2','3','4','5']], [['stages']]]]
        # network_structure = [['encoder','decoder'],[[['stages', 'stages', 'stages', 'stages', 'stages'], ['0','1','2','3','4','5']], [['stages']]]]
        network_structure = [['decoder'],[[['stages']]]]
        
        if not all(i[0][0] == 'stages' for i in network_structure[1]):
            raise NotImplementedError("Only implemented for 'stages' network structure.") 
            
        #Lets only checkpoint the decoder for now as the skip connections
        #will make it more memory intensive.

        #Lets assign a set of "levels" which we will permit checkpointing within. Even then, we will
        #not probably checkpoint the entire block, just the most memory intensive activations. 
        
        #Now at the intra-level we can assign blocks which we will checkpoint.

        #Lets assign the blocks within a "level" which can be checkpointed also.
        # encoder_blocks = ['blocks', ['conv1', 'conv2', 'conv3', 'skip']] #nnu-net uses the same overall naming convention within
        encoder_blocks = ['blocks']

        decoder_blocks = ['convs', ['conv']] #nnu-net uses the same overall naming convention within
        #a block, so we can use this to iterate over the blocks and checkpoint specific layers.
        if encoder_blocks[0] != 'blocks' or decoder_blocks[0] != 'convs':
            raise NotImplementedError("Only implemented for 'blocks' and 'convs' intra-level structure.")
        

        #it is not consistent between the encoder and decoder though, so we need to be careful.
        
        for name, module in model.named_modules():
            parts = name.split('.') #Lets split by dot to get the module hierarchy.
            #We will filter valid modules by order of hierarchy.
            #we create a list which tracks at each level if we have matched a module.
            if len(parts) < len(network_structure):
                continue #Too short to match.
            #Now lets iterate over the encoder/decoder structure.
            for idx, struct in enumerate(network_structure[0]):
                if struct == 'decoder':
                    dummy_network_structures = [[struct] + network_structure[1][idx][0]]
                elif struct == 'encoder':
                    # This was for looking within each block within the stage.
                    dummy_network_structures = [[struct] + [i, j] for i, j in zip(network_structure[1][idx][0], network_structure[1][idx][1])]
                   
                else:
                    raise ValueError("Unknown network structure for checkpointing.")
                
                for dummy_network_structure in dummy_network_structures:
                    if len(parts) < len(dummy_network_structure):
                        continue #Too short to match.
                    structure_checker = [parts[i] == dummy_network_structure[i] for i in range(len(dummy_network_structure))]
                    if not all(structure_checker):
                        continue
                    
                    #If structure matches now we look at intra level
                    if dummy_network_structure[0] == 'decoder':
                        look_here = decoder_blocks
                    elif dummy_network_structure[0] == 'encoder':
                        look_here = encoder_blocks
                    else:
                        raise ValueError("Unknown network structure for checkpointing.")
                    blockparts = parts[len(dummy_network_structure):]

                    if dummy_network_structure[0] == 'decoder':
                        if len(blockparts) != 2 * len(look_here): #Stages have indexing, and so do the blocks within
                            #each level.
                            continue #Not the right level to match, either too coarse or too granular. 
                    elif dummy_network_structure[0] == 'encoder':
                        #Here we actually are going to checkpoint entire blocks so we need to 
                        #adjust our granularity.
                        if len(blockparts) != len(look_here) + 1: #Stages have indexing, and so do the blocks within
                            #each level. We are going to be looking at entire blocks here, nothing more granular.
                            continue #Not the right level to match, either too coarse or too granular.

                    # #Lets look past indexing of the stages.
                    # if blockparts[1] != look_here[0]:
                    #     continue
                    # #Within the block, we also have indexing of conv blocks.
                    # if blockparts[2] != look_here[1]:
                    #     continue
                    #We wrap both checks into one for loop.
                    if struct == 'encoder':
                        #This old tracker was for checkpointing within each block within a stage.
                        # intra_tracker = [blockparts[i * 2] in look_here[i] for i in range(len(look_here))]
                        intra_tracker = [blockparts[0] == look_here[0]] #We checkpoint entire blocks here. 
                    elif struct == 'decoder':
                        intra_tracker = [blockparts[i * 2 + 1] in look_here[i] for i in range(len(look_here))]
                    if not all(intra_tracker):
                        continue
                    else:
                        #We have a match, we will checkpoint this module's forward pass.
                        try:
                            block = model.get_submodule(name)
                            # # Replace the block with a checkpointed version
                            # parent_path, block_name = name.rsplit('.', 1)
                            # parent = model.get_submodule(parent_path) if parent_path else model
                            # setattr(parent, block_name, CheckpointedModule(block, use_reentrant=False))
                            # print(f"✓ Checkpointed {name}")
                            # Capture ORIGINAL forward before ANY replacement
                            original_forward = block.__class__.forward.__get__(block, block.__class__)
                            
                            # Now replace with checkpointed version
                            def make_checkpointed(orig, block_name):
                                def forward_fn(*args, **kwargs):
                                    print(f"[CHECKPOINT] Running checkpointed {block_name}")
                                    return checkpoint(orig, *args, use_reentrant=False, **kwargs)
                                return forward_fn
                            
                            block.forward = make_checkpointed(original_forward, name)
                        except Exception as e:
                            print(f"✗ Failed to checkpoint {name}: {e}")

            
        return model
    
    def apply_forward_checkpoint_test_orig(self, model):
        orig_forward = model.forward
        def wrapped_forward(*args):
            return checkpoint(partial(orig_forward), *args, use_reentrant=False)  # kwargs avoided; use partial if needed
        model.forward = wrapped_forward
        return model 
    def build_network_architecture(self, device:torch.device) -> torch.nn.Module:
        model = super().build_network_architecture(device=device).to(device=device)

        #Normally we look at the discrepancy between existing and current kwargs to determine how to configure,
        #but this is just a dummy which adds batchnorm layers for debugging.
        
        # First, freeze ALL parameters in the entire model
        for param in model.parameters():
            param.requires_grad = False
        
        # Then, unfreeze only the affine parameters (weight and bias) of InstanceNorm3d layers
        for name, module in model.named_modules():
            if isinstance(module, nn.InstanceNorm3d):
                if module.affine:
                    module.weight.requires_grad = True
                    module.bias.requires_grad = True
                else:
                    raise ValueError("InstanceNorm3d layer does not have affine parameters to train.") 
            #Now lets unfreeze the conv blocks too. We will restrict this to the stem and decoder seg layer conv blocks for now, as these are the most relevant for adaptation and also to limit the number of trainable parameters.
            if isinstance(module, nn.Conv3d) and ('stem' in name or 'seg_layers' in name): 
                module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True

        #Lets try and checkpoint some operations so that we don't run out of VRAM....
        #slower but at least it won't BREAK. This will need to overwrite the forward function of
        #the prior model though, which is annoying. 
        # model = self.apply_forward_checkpoint(model)
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
    'nnInteractiveUNetFrozenDebugging': make_factory(nnInteractiveUNetFrozenDebugging),
    'nnInteractiveUNetTrainNorm': make_factory(nnInteractiveUNetInstanceNorm),
    'nnInteractiveUNetTrainConv': make_factory(nnInteractiveUNetConv),
    'nnInteractiveUNetTrainConvNorm': make_factory(nnInteractiveUNetConvNorm), #This is the same as nnInteractiveUNetInstanceNorm for now, since we are unfreezing both conv and norm layers in that class.
    # 'nnInteractiveUNetTrainFILM': make_factory(nnInteractiveUNetFILM),
}
