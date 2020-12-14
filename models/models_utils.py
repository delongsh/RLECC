import torch
import torch.nn as nn

def transfer_params(source_model, target_model):
    source_model_state_dict = source_model.state_dict()
    target_model_state_dict = target_model.state_dict()
    source_model_state_dict = {k: v for k, v in source_model_state_dict.items() if k in target_model_state_dict}
    target_model_state_dict.update(source_model_state_dict)
    target_model.load_state_dict(target_model_state_dict)

    return target_model