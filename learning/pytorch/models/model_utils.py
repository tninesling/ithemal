import torch
from typing import Any, Dict

def dump_shared_params(module):
    # type: (torch.nn.Module) -> Dict[str, Any]
    shared_params = {}
    
    for name, param in module.named_parameters():
        # Create a shared tensor
        shared_tensor = param.data.detach().clone().share_memory_()
        
        # Store the tensor directly
        shared_params[name] = shared_tensor
        
    return shared_params

def load_shared_params(module, params):
    # type: (torch.nn.Module, Dict[str, Any]) -> None
    
    for name, param in module.named_parameters():
        if name in params:
            shared_tensor = params[name]
            param.data = shared_tensor
