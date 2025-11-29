from absl import logging
from pynvml import *
import haiku as hk
import numpy as np
import pickle
import os


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
    if model_type == "mpnn" or model_type == "gat" or model_type == "triplet_mpnn":
        key_word = "linear"
    elif model_type == "edge_t":
        key_word = "ET_Layer"
    else:
        raise ValueError("model_type not defined")
    
    if ('processor' in key) and (key_word in key) and "layer_norm" not in key:
        key_word_index = key.index(key_word)
        if key[(key_word_index + len(key_word))%len(key)] == "_":
            cur_layer = int(key[key_word_index + len(key_word)+1])
        else:
            cur_layer = 0
        
        if int(cur_layer) >= layer_threshold:
            return True
        
    return False

def assign_to_model_parameters(model, params=None, layer=0, model_type = "mpnn",
                               remap_algo_from_list=None, remap_algo_to_list=None):
    """Restore model from `file_name`.

    Args:
      model: model object with `checkpoint_path` and `params` attributes.
      params: optional flattened 1-D numpy array of processor-weight updates to
        apply to the checkpoint parameters before merging. If None, the
        checkpoint params are merged directly.
      layer: layer threshold passed to `filter_layers` to select which
        processor weights to update from `params`.
      model_type: passed to `filter_layers`.
      remap_algo_from: if not None, remap any restored keys containing
        `encoders_decoders` and `algo_{remap_algo_from}` to use
        `algo_{remap_algo_to}` instead (useful when decoder indices differ).
      remap_algo_to: target algorithm index for remapping.
    """
    path = os.path.join(model.checkpoint_path, 'best.pkl')
    with open(path, 'rb') as f:
      restored_state = pickle.load(f)
      restored_params = restored_state['params']

      # Optionally remap encoder/decoder keys.
      # Support either single remap (remap_algo_from -> remap_algo_to) or
      # list remaps (remap_algo_from_list -> remap_algo_to_list)
      if remap_algo_from_list is not None:
        if remap_algo_to_list is None or len(remap_algo_from_list) != len(remap_algo_to_list):
          raise ValueError('remap_algo_from_list and remap_algo_to_list must be same-length lists')
        remapped_params = {}
        # Build mapping dict of string tokens
        mapping = {f"algo_{int(a)}": f"algo_{int(b)}" for a, b in zip(remap_algo_from_list, remap_algo_to_list)}
        for key, sub in restored_params.items():
          new_key = key
          if "encoders_decoders" in key:
            # find if any from-token is present; if so, remap to corresponding to-token
            matched = False
            for from_tok, to_tok in mapping.items():
              if from_tok in key:
                new_key = key.replace(from_tok, to_tok)
                logging.info('Remapping checkpoint key %s -> %s', key, new_key)
                matched = True
                break
            # Only include encoder/decoder keys that matched a mapping (drop others)
            if matched:
              remapped_params[new_key] = sub
          else:
            # keep non-encoder/decoder keys (e.g., processor params)
            remapped_params[new_key] = sub
        restored_params = remapped_params
      
      # If a flattened params vector is provided, apply updates to a fresh
      # `new_params` dict built from `restored_params` (do not mutate the
      # checkpoint dict in-place).
      if params is not None:
        cur_len = 0
        new_params = {}
        for key in restored_params.keys():
          new_params[key] = {}
          for param in restored_params[key]:
            cur_val = restored_params[key][param]
            updated_val = cur_val
            if 'processor' in key and "layer_norm" not in key:
              cur_dim = np.prod(cur_val.shape)
              cur_param = params[cur_len:cur_len+cur_dim]
              if cur_param.size != cur_dim:
                raise ValueError(f"Provided params length mismatch at key={key}, param={param}: "
                                 f"expected {cur_dim} elements, got {cur_param.size}")
              if filter_layers(key, layer, model_type=model_type):
                print(f"Assigning {key} {param} to model")
                updated_val = cur_val + np.reshape(cur_param, cur_val.shape)
              cur_len += cur_dim
            new_params[key][param] = updated_val

        if cur_len != params.size:
          logging.warning("Flattened params length (%d) does not match consumed dim (%d)", params.size, cur_len)

        model.params = hk.data_structures.merge(model.params, new_params)
      else:
        # No flattened updates provided — merge the (possibly remapped)
        # checkpoint parameters directly.
        model.params = hk.data_structures.merge(model.params, restored_params)

      # Reinitialize optimizer state for the (possibly remapped) parameters.
      # Restoring a saved optimizer state is only safe when the checkpoint
      # opt_state structure exactly matches the current model; otherwise
      # JAX treedef mismatches will occur. Reinitialize to avoid that.
      try:
        model.opt_state = model.opt.init(model.params)
        logging.info('Merged checkpoint params; optimizer state reinitialized.')
      except Exception as e:
        logging.warning('Failed to reinitialize optimizer state: %s', e)
        # As a fallback, avoid setting opt_state to incompatible checkpoint state.

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