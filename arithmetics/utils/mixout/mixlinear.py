##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Cheolhyoung Lee
## Department of Mathematical Sciences, KAIST
## Email: cheolhyoung.lee@kaist.ac.kr
## Implementation of mixout from https://arxiv.org/abs/1909.11299
## "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional
from collections import OrderedDict
from .functional import mixout
from copy import deepcopy
from transformers.pytorch_utils import Conv1D



class MixLinear(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']
    # If target is None, nn.Sequential(nn.Linear(m, n), MixLinear(m', n', p)) 
    # is equivalent to nn.Sequential(nn.Linear(m, n), nn.Dropout(p), nn.Linear(m', n')).
    # If you want to change a dropout layer to a mixout layer, 
    # you should replace nn.Linear right after nn.Dropout(p) with Mixout(p) 
    def __init__(self, 
                in_features:int, 
                out_features:int, 
                bias:bool=True, 
                target:Optional["OrderedDict[str, torch.Tensor]"]=None, 
                p:float=0.0) -> None:

        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.target = target
        self.p = p

        if self.p < 0 or self.p > 1:
            raise ValueError(f"A mix probability of mixout has to be between 0 and 1,  but got {self.p}")
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        return F.linear(input, mixout(self.weight, self.target, 
                                      self.p, self.training), self.bias)

    def extra_repr(self):
        type_ = 'drop' if self.target is None else 'mix'
        type_ += "out" 
        return f'{type_}={self.p}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    
class MixConv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx,
                target:Optional["OrderedDict[str, torch.Tensor]"]=None, 
                p:float=0.0):
        super(MixConv1D, self).__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

        self.target = target
        self.p = p
        if self.p < 0 or self.p > 1:
            raise ValueError(f"A mix probability of mixout has to be between 0 and 1,  but got {self.p}")

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), mixout(self.weight, self.target, 
                                      self.p, self.training))
        x = x.view(size_out)
        return x

    def extra_repr(self):
        type_ = 'drop' if self.target is None else 'mix'
        type_ += "out" 
        return f'{type_}={self.p}, in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, bias={self.bias is not None}'
    

def replace_layer_for_mixout(module: nn.Module, mixout_prob: float) -> nn.Module:
    '''
    Replaces a single layer with the correct layer for use with Mixout.
    If module is nn.Dropout, replaces it with a Dropout where p = 0.
    If module is nn.Linear, replaces it with a MixLinear where p(mixout) = mixout_prob.
    In all other cases, returns the module unchanged.
    
        params:
            module (nn.Module)    : a module to replace for Mixout
            mixout_prob (float)   : the desired Mixout probability
        
        returns:
            module (nn.Module)    : the module set up for use with Mixout
    '''
    if isinstance(module, nn.Dropout):
        return nn.Dropout(0)
    elif isinstance(module, nn.Linear):
        target_state_dict   = deepcopy(module.state_dict())
        bias                = True if module.bias is not None else False
        new_module          = MixLinear(
                                module.in_features,
                                module.out_features,
                                bias,
                                target_state_dict['weight'],
                                mixout_prob
                            )
        new_module.load_state_dict(target_state_dict)
        return new_module
    elif isinstance(module, Conv1D):
        target_state_dict   = deepcopy(module.state_dict())
        new_module          = MixConv1D(
                                module.weight.shape[1],
                                module.weight.shape[0],
                                target_state_dict['weight'],
                                mixout_prob
                            )
        new_module.load_state_dict(target_state_dict)
        return new_module
    else:
        return module

def recursive_setattr(obj: 'any', attr: str, value: 'any') -> None:
    '''
    Recursively sets attributes for objects with children.
    
        params:
            obj (any)   : the object with children whose attribute is to be set
            attr (str)  : the (nested) attribute of the object, with levels indicated by '.'
                            for instance attr='attr1.attr2' sets the attr2 of obj.attr1 to
                            the passed value
            value (any) : what to set the attribute to
    '''
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)
