from absl import logging
from pynvml import *
import haiku as hk
import numpy as np
import pickle


def print_gpu_utilization():
    nvmlInit()
    memory = 0
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    # print(info.used, info.total)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    memory += info.used//1024**2

    handle = nvmlDeviceGetHandleByIndex(1)
    info = nvmlDeviceGetMemoryInfo(handle)
    # print(info.used, info.total)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    memory += info.used//1024**2
    print(f"Total GPU memory occupied: {memory} MB.")

def filter_layers(key, layer_threshold, model_type = "mpnn"):
    '''
    return True if the key is located beyond the layer_threshold
    '''
    if model_type == "mpnn":
        key_word = "linear"
    elif model_type == "edge_t":
        key_word = "ET_Layer"
    else:
        raise ValueError("model_type must be either 'mpnn' or 'edge_t'")
    
    if ('processor' in key) and (key_word in key) and "layer_norm" not in key:
        key_word_index = key.index(key_word)
        if key[(key_word_index + len(key_word))%len(key)] == "_":
            cur_layer = int(key[key_word_index + len(key_word)+1])
        else:
            cur_layer = 0
        
        if int(cur_layer) >= layer_threshold:
            return True
        
    return False

def assign_to_model_parameters(model, params=None, layer=0):
    """Restore model from `file_name`."""
    path = os.path.join(model.checkpoint_path, 'best.pkl')
    with open(path, 'rb') as f:
      restored_state = pickle.load(f)
      restored_params = restored_state['params']
      
      if params is not None:
        cur_len = 0
        for key in restored_params.keys():
            for param in restored_params[key]:
              if 'processor' in key and "layer_norm" not in key:
                cur_dim = np.prod(restored_params[key][param].shape)
                cur_param = params[cur_len:cur_len+cur_dim]
                if filter_layers(key, layer):
                  restored_params[key][param] += np.reshape(cur_param, restored_params[key][param].shape)
                else: pass; # skip: not assigning the update parameters to the model
                cur_len += cur_dim

      model.params = hk.data_structures.merge(model.params, restored_params)
      model.opt_state = restored_state['opt_state']

def restore_model(model, file_name: str, change_algo_index=0):
    """Restore model from `file_name`."""
    path = os.path.join(model.checkpoint_path, file_name)
    new_params = {}
    with open(path, 'rb') as f:
      restored_state = pickle.load(f)
      restored_params = restored_state['params']
      if change_algo_index is not None:
        for key in restored_params:
          if "processor" in key:
            new_params[key] = restored_params[key]
          if "encoders_decoders" in key and ("algo_{}".format(int(change_algo_index)) in key):
            new_params[key.replace("algo_{}".format(int(change_algo_index)), "algo_{}".format(0))] = restored_params[key]
      else:
        new_params = restored_params
      model.params = hk.data_structures.merge(model.params, new_params)
    logging.info('Model restored from %s', path)

def get_pretrained_model_weights(model, layer=0):
    path = os.path.join(model.checkpoint_path, 'best.pkl')
    with open(path, 'rb') as f:
      restored_state = pickle.load(f)
      restored_params = restored_state['params']
      
      model_params = []
      for key in restored_params.keys():
          for param in restored_params[key]:
            if 'processor' in key and "layer_norm" not in key:
                model_params.append(restored_params[key][param].flatten())
      return np.concatenate(model_params)