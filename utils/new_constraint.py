import torch
from torch.nn import BatchNorm2d

def _frob_norm(w):
    return torch.sum(torch.pow(w, 2.0))

class OldTopKFrobeniusConstraint(object): 

    def __init__(self, model_type, max_k, state_dict = None,
                 excluding_key = [], including_key = []) -> None:
        self.model_type = model_type
        self.max_k = max_k
        self.state_dict = state_dict
        self.excluding_key = excluding_key
        self.including_key = including_key

    def __call__(self, module):
        if type(module) == self.model_type:
            param_dict = {}
            score_dict = {}
            for name, param in module.named_parameters():
                if "bias" in name:
                    continue
            
                if (len(self.excluding_key)>0):
                    for key in self.excluding_key:
                        if key in name:
                            continue
                
                for key in self.including_key:
                    if key in name:
                        param_dict[name] = param

                        w = param.data
                        grad = param.grad.data

                        scores = torch.sign((w - self.state_dict[name])*grad)*torch.abs(grad) # (w - self.state_dict[name])*grad
                        score_dict[name] = scores
                        break

            # distribute constraint budget according to scores         
            total_score = torch.cat([torch.flatten(v) for v in score_dict.values()])
            total_param = torch.cat([torch.flatten(v.data - self.state_dict[key]) for key, v in param_dict.items()])
            current_norm =  _frob_norm(total_param)
            if current_norm < self.max_k: return

            k = int(len(total_param)*0.01); interval = int(len(total_param)*0.005)
            _, indices = torch.topk(total_score, k, sorted=False)
            while _frob_norm(total_param[indices]) < current_norm - self.max_k:
                k += interval
                _, indices = torch.topk(total_score, k, sorted=False)
            k -= interval
            _, indices = torch.topk(total_score, k, sorted=True)

            # make the params correspond to indices to zero
            masks = torch.zeros_like(total_param, dtype=torch.bool)
            masks[indices] = 1
            mask_dict = {}; cur_len = 0
            for name, param in param_dict.items():
                tmp_mask = masks[cur_len:cur_len+param.numel()].reshape(param.shape)
                mask_dict[name] = tmp_mask
                cur_len += param.numel()
            
            # apply mask
            for name, param in param_dict.items():
                tmp_mask = mask_dict[name]
                param.data[tmp_mask] = self.state_dict[name][tmp_mask]

class UniformFrobeniusConstraint(object): 

    def __init__(self, model_type, max_k, state_dict = None,
                 excluding_key = [], including_key = []) -> None:
        self.model_type = model_type
        self.max_k = max_k
        self.state_dict = state_dict
        self.excluding_key = excluding_key
        self.including_key = including_key

    def __call__(self, module):
        if type(module) == self.model_type:
            param_dict = {}
            score_dict = {}
            for name, param in module.named_parameters():
                if "bias" in name:
                    continue
            
                if (len(self.excluding_key)>0):
                    for key in self.excluding_key:
                        if key in name:
                            continue
                
                for key in self.including_key:
                    if key in name:
                        param_dict[name] = param

                        w = param.data
                        break

            # distribute constraint budget according to scores         
            total_param = torch.cat([torch.flatten(v.data - self.state_dict[key]) for key, v in param_dict.items()])
            current_norm =  _frob_norm(total_param)
            if current_norm < self.max_k: return

            # rescale proportionally to current norm/max norm
            for name, param in param_dict.items():
                param.data = (param.data - self.state_dict[name]) * (self.max_k/current_norm)**0.5 + self.state_dict[name]