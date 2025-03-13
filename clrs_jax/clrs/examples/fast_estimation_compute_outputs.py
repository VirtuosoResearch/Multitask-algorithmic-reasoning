"""Run training of one or more algorithmic tasks from CLRS."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import functools
import shutil
from typing import Any, Dict, List

from absl import app
from absl import flags
from absl import logging
import clrs
import jax
import numpy as np
import requests
import tensorflow as tf
import wandb
import pickle
import haiku as hk

import time
from clrs import utils

flags.DEFINE_list('algorithms', ['dijkstra'], 'Which algorithms to run.')
flags.DEFINE_list('train_lengths', ['16'],
                  'Which training sizes to use. A size of -1 means '
                  'use the benchmark dataset.')
flags.DEFINE_integer('length_needle', -8,
                     'Length of needle for training and validation '
                     '(not testing) in string matching algorithms. '
                     'A negative value randomizes the length for each sample '
                     'between 1 and the opposite of the value. '
                     'A value of 0 means use always 1/4 of the length of '
                     'the haystack (the default sampler behavior).')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_boolean('random_pos', True,
                     'Randomize the pos input common to all algos.')
flags.DEFINE_boolean('enforce_permutations', True,
                     'Whether to enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True,
                     'Whether to change pred_h hints into pred inputs.')
flags.DEFINE_integer('batch_size', 4, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', False,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 16,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('train_steps', 10000, 'Number of training iterations.')
flags.DEFINE_integer('eval_every', 50, 'Evaluation frequency (in steps).')
flags.DEFINE_integer('test_every', 500, 'Test frequency (in steps).')

flags.DEFINE_integer('hidden_size', 192,
                     'Number of hidden units of the model.')
flags.DEFINE_integer('num_layers', 3,
                     'Number of layers of the network.')
flags.DEFINE_integer('nb_heads', 12, 'Number of heads for GAT processors')
flags.DEFINE_integer('nb_msg_passing_steps', 1,
                     'Number of message passing steps to run per hint.')
flags.DEFINE_float('learning_rate', 2.5e-4, 'Learning rate to use.')
flags.DEFINE_float('grad_clip_max_norm', 1.0,
                   'Gradient clipping by norm. 0.0 disables grad clipping')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_float('attention_dropout_prob', 0.0, 'Dropout rate in the attention heads to use.')
flags.DEFINE_enum('activation', 'relu', ['relu', 'gelu'], 'Type of activation function to use.')
flags.DEFINE_integer('use_graph_fts', 1,
                     'Whether to use the graph features.')
flags.DEFINE_integer('ood_val', 0,
                     'Whether to apply bigger graphs for the val. samples.')
flags.DEFINE_float('hint_teacher_forcing', 0.0,
                   'Probability that ground-truth teacher hints are encoded '
                   'during training instead of predicted hints. Only '
                   'pertinent in encoded_decoded modes.')
flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'How should hints be used? Note, each mode defines a '
                  'separate task, with various difficulties. `encoded_decoded` '
                  'requires the model to explicitly materialise hint sequences '
                  'and therefore is hardest, but also most aligned to the '
                  'underlying algorithmic rule. Hence, `encoded_decoded` '
                  'should be treated as the default mode for our benchmark. '
                  'In `decoded_only`, hints are only used for defining '
                  'reconstruction losses. Often, this will perform well, but '
                  'note that we currently do not make any efforts to '
                  'counterbalance the various hint losses. Hence, for certain '
                  'tasks, the best performance will now be achievable with no '
                  'hint usage at all (`none`).')
flags.DEFINE_enum('hint_repred_mode', 'soft', ['soft', 'hard', 'hard_on_eval'],
                  'How to process predicted hints when fed back as inputs.'
                  'In soft mode, we use softmaxes for categoricals, pointers '
                  'and mask_one, and sigmoids for masks. '
                  'In hard mode, we use argmax instead of softmax, and hard '
                  'thresholding of masks. '
                  'In hard_on_eval mode, soft mode is '
                  'used for training and hard mode is used for evaluation.')
flags.DEFINE_boolean('norm_first_att', True,
                     'Whether to normalize the input to the attention layers.')
flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_integer('nb_triplet_fts', 8,
                     'How many triplet features to compute?')
flags.DEFINE_boolean('use_projection', False,
                      'Whether to use a projection layer after the processor.')
flags.DEFINE_integer('projection_dim', 16, 'Dimension of the projection layer.')

flags.DEFINE_enum('encoder_init', 'xavier_on_scalars',
                  ['default', 'xavier_on_scalars'],
                  'Initialiser to use for the encoders.')
flags.DEFINE_enum('processor_type', 'edge_t',
                  ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
                   'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
                   'gat', 'gatv2', 'gat_full', 'gatv2_full',
                   'gpgn', 'gpgn_mask', 'gmpnn',
                   'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn', 'edge_t', 'branching_edge_t'],
                  'Processor type to use as the network P.')
flags.DEFINE_enum('node_readout', 'diagonal',
                  ['diagonal', 'sophisticated'],
                  'Method to extract the node features from transformer output.')

flags.DEFINE_string('checkpoint_path', './saved',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', './data/CLRS30',
                    'Path in which dataset is stored.')
flags.DEFINE_string('wandb_entity', None,
                    'Name of `wandb` entity.')
flags.DEFINE_string('wandb_project', None,
                    'Name of `wandb` project.')
flags.DEFINE_string('wandb_name', None,
                    'Name of `wandb` run.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')

flags.DEFINE_string('load_checkpoint_path', 'test',
                    'Path from which to load the checkpoint.')
flags.DEFINE_integer('gradient_projection_seed', 0, 'Seed')
flags.DEFINE_integer('gradient_projection_dim', 400, 'Dimension of the gradient projection')
flags.DEFINE_integer('change_algo_index', 0, 'Index of the algorithm to change the checkpoint')

flags.DEFINE_integer('runs', 10, "Runs")
flags.DEFINE_integer('layer', 0, "Runs")
flags.DEFINE_float('perturb_ratio', 0.01, 'Learning rate for the gradient projection')


flags.DEFINE_boolean('use_branching_structure', False,
                     'Whether to define the branching structure of the model.')
flags.DEFINE_string('branching_structure_dir', None,
                    'Path to the directory where the branching structure is stored.')


FLAGS = flags.FLAGS


PRED_AS_INPUT_ALGOS = [
    'binary_search',
    'minimum',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'naive_string_matcher',
    'kmp_matcher',
    'jarvis_march']


def unpack(v):
  try:
    return v.item()  # DeviceArray
  except (AttributeError, ValueError):
    return v


def _iterate_sampler(sampler, batch_size):
  while True:
    yield sampler.next(batch_size)


def _maybe_download_dataset(dataset_path):
  """Download CLRS30 dataset if needed."""
  dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
  if os.path.isdir(dataset_folder):
    logging.info('Dataset found at %s. Skipping download.', dataset_folder)
    return dataset_folder
  logging.info('Dataset not found in %s. Downloading...', dataset_folder)

  clrs_url = clrs.get_dataset_gcp_url()
  request = requests.get(clrs_url, allow_redirects=True)
  clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
  os.makedirs(dataset_folder)
  open(clrs_file, 'wb').write(request.content)
  shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
  os.remove(clrs_file)
  return dataset_folder


def make_sampler(length: int,
                 rng: Any,
                 algorithm: str,
                 split: str,
                 batch_size: int,
                 multiplier: int,
                 randomize_pos: bool,
                 enforce_pred_as_input: bool,
                 enforce_permutations: bool,
                 chunked: bool,
                 chunk_length: int,
                 sampler_kwargs: Dict[str, Any]):
  """Create a sampler with given options.

  Args:
    length: Size of samples (i.e., number of nodes in the graph).
      A length of -1 will mean that the benchmark
      dataset (for the given split) is used. Positive sizes will instantiate
      samplers of the corresponding size.
    rng: Numpy random state.
    algorithm: The name of the algorithm to sample from.
    split: 'train', 'val' or 'test'.
    batch_size: Samples per batch.
    multiplier: Integer multiplier for the number of samples in the dataset,
      only used for positive sizes. Negative multiplier means infinite samples.
    randomize_pos: Whether to randomize the `pos` input.
    enforce_pred_as_input: Whether to convert fixed pred_h hints to inputs.
    enforce_permutations: Whether to enforce permutation pointers.
    chunked: Whether to chunk the dataset.
    chunk_length: Unroll length of chunks, if `chunked` is True.
    sampler_kwargs: Extra args passed to the sampler.
  Returns:
    A sampler (iterator), the number of samples in the iterator (negative
    if infinite samples), and the spec.
  """
  if length < 0:  # load from file
    dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
    sampler, num_samples, spec = clrs.create_dataset(folder=dataset_folder,
                                                     algorithm=algorithm,
                                                     batch_size=batch_size,
                                                     split=split)
    sampler = sampler.as_numpy_iterator()
  else:
    num_samples = clrs.CLRS30[split]['num_samples'] * multiplier
    sampler, spec = clrs.build_sampler(
        algorithm,
        seed=rng.randint(2**32),
        num_samples=num_samples,
        length=length,
        **sampler_kwargs,
        )
    sampler = _iterate_sampler(sampler, batch_size)

  if randomize_pos:
    sampler = clrs.process_random_pos(sampler, rng)
  if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
    spec, sampler = clrs.process_pred_as_input(spec, sampler)
  spec, sampler = clrs.process_permutations(spec, sampler, enforce_permutations)
  if chunked:
    sampler = clrs.chunkify(sampler, chunk_length)
  # Note: sampler is as an generator
  return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, **kwargs):
  """Create a sampler with cycling sample sizes."""
  ss = []
  tot_samples = 0
  for length in sizes:
    sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
    ss.append(sampler)
    tot_samples += num_samples

  def cycle_samplers():
    while True:
      for s in ss:
        yield next(s)
  return cycle_samplers(), tot_samples, spec


def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
  """Collect batches of output and hint preds and evaluate them."""
  processed_samples = 0
  preds = []
  outputs = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    preds.append(cur_preds)
    processed_samples += batch_size
  outputs = _concat(outputs, axis=0)
  preds = _concat(preds, axis=0)
  out = clrs.evaluate(outputs, preds)
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}


def create_samplers(rng, train_lengths: List[int]):
  """Create all the samplers."""
  train_samplers = []
  val_samplers = []
  val_sample_counts = []
  test_samplers = []
  test_sample_counts = []
  spec_list = []
  is_graph_fts_avail = []

  for algo_idx, algorithm in enumerate(FLAGS.algorithms):
    # Make full dataset pipeline run on CPU (including prefetching).
    with tf.device('/cpu:0'):

      if algorithm in ['naive_string_matcher', 'kmp_matcher']:
        # Fixed haystack + needle; variability will be in needle
        # Still, for chunked training, we maintain as many samplers
        # as train lengths, since, for each length there is a separate state,
        # and we must keep the 1:1 relationship between states and samplers.
        max_length = max(train_lengths)
        if max_length > 0:  # if < 0, we are using the benchmark data
          max_length = (max_length * 5) // 4
        train_lengths = [max_length]
        if FLAGS.chunked_training:
          train_lengths = train_lengths * len(train_lengths)

      logging.info('Creating samplers for algo %s', algorithm)

      p = tuple([0.1 + 0.1 * i for i in range(9)])
      if p and algorithm in ['articulation_points', 'bridges',
                             'mst_kruskal', 'bipartite_matching']:
        # Choose a lower connection probability for the above algorithms,
        # otherwise trajectories are very long
        p = tuple(np.array(p) / 2)
      length_needle = FLAGS.length_needle
      sampler_kwargs = dict(p=p, length_needle=length_needle)
      if length_needle == 0:
        sampler_kwargs.pop('length_needle')

      common_sampler_args = dict(
          algorithm=FLAGS.algorithms[algo_idx],
          rng=rng,
          enforce_pred_as_input=FLAGS.enforce_pred_as_input,
          enforce_permutations=FLAGS.enforce_permutations,
          chunk_length=FLAGS.chunk_length,
          )

      train_args = dict(sizes=train_lengths,
                        split='train',
                        batch_size=FLAGS.batch_size,
                        multiplier=-1,
                        randomize_pos=FLAGS.random_pos,
                        chunked=FLAGS.chunked_training,
                        sampler_kwargs=sampler_kwargs,
                        **common_sampler_args)
      train_sampler, _, spec = make_multi_sampler(**train_args)

      mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
      if FLAGS.ood_val:
          val_size = [32]
      else:
          val_size = [np.amax(train_lengths)]
      val_args = dict(sizes=val_size,
                      split='val',
                      batch_size=32,
                      multiplier=2 * mult,
                      randomize_pos=FLAGS.random_pos,
                      chunked=False,
                      sampler_kwargs=sampler_kwargs,
                      **common_sampler_args)
      val_sampler, val_samples, spec = make_multi_sampler(**val_args)

      test_args = dict(sizes=[-1],
                       split='test',
                       batch_size=32,
                       multiplier=2 * mult,
                       randomize_pos=False,
                       chunked=False,
                       sampler_kwargs={},
                       **common_sampler_args)
      test_sampler, test_samples, spec = make_multi_sampler(**test_args)

    spec_list.append(spec)
    train_samplers.append(train_sampler)
    val_samplers.append(val_sampler)
    val_sample_counts.append(val_samples)
    test_samplers.append(test_sampler)
    test_sample_counts.append(test_samples)
    input_spec = list(filter(lambda x: x[0] in ['input', 'hint'], spec.values()))
    graph_fts_exists = any('graph' in value for value in input_spec) 
    is_graph_fts_avail.append(graph_fts_exists)

  return (train_samplers,
          val_samplers, val_sample_counts,
          test_samplers, test_sample_counts,
          spec_list, is_graph_fts_avail)


def load_branching_structure(branching_structure_dir, algorithms, num_layers):
  """Load the branching structure of the model."""
  branching_structure = []
  for _ in range(num_layers):
    branching_structure.append({i: 0 for i in range(len(algorithms))}) 
  num_groups = [0]*num_layers

  with open(os.path.join("tree_configs", branching_structure_dir+".txt"), 'r') as f:
    for line in f.readlines():
      line = line.strip()
      layer, task_group = line.split(":")
      layer = int(layer); task_group = task_group.strip().split(" ")
      for algo in task_group:
        branching_structure[layer].update({algorithms.index(algo): num_groups[layer]})
      num_groups[layer] += 1
  print("Branching structure: ", branching_structure)
  return branching_structure

def main(unused_argv):
  if FLAGS.hint_mode == 'encoded_decoded':
    encode_hints = True
    decode_hints = True
  elif FLAGS.hint_mode == 'decoded_only':
    encode_hints = False
    decode_hints = True
  elif FLAGS.hint_mode == 'none':
    encode_hints = False
    decode_hints = False
  else:
    raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

  train_lengths = [int(x) for x in FLAGS.train_lengths]

  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2**32))

  # Create samplers. Note: This is the dataset
  (train_samplers,
   val_samplers, val_sample_counts,
   test_samplers, test_sample_counts,
   spec_list, is_graph_fts_avail) = create_samplers(rng, train_lengths)

  if FLAGS.wandb_project:
    if FLAGS.ood_val:
        val_size = [32]
    else:
        val_size = [np.amax(train_lengths)]
    config = {
      "processor": FLAGS.processor_type,
      "algorithm": FLAGS.algorithms[0],
      "activation": FLAGS.activation,
      "step_nums": FLAGS.train_steps,
      "lr": FLAGS.learning_rate,
      "batch_size": FLAGS.batch_size,
      "nb_heads": FLAGS.nb_heads,
      "num_layers": FLAGS.num_layers,
      "hidden_size": FLAGS.hidden_size,
      "attention_dropout_rate": FLAGS.attention_dropout_prob,
      "norm_first_att": FLAGS.norm_first_att,
      "seed_param": FLAGS.seed,
      "readout": FLAGS.node_readout,
      "ood_val": FLAGS.ood_val,
      "val_size": val_size,
    }
    wandb.init(
      entity=FLAGS.wandb_entity,
      project=FLAGS.wandb_project,
      name=FLAGS.wandb_name,
      config={"dataset": "clrs30", **dict(config)},
    )

  if FLAGS.use_branching_structure:
    branching_structure = load_branching_structure(FLAGS.branching_structure_dir, FLAGS.algorithms, FLAGS.num_layers)
  else:
    branching_structure = None

  processor_factory = clrs.get_processor_factory(
      FLAGS.processor_type,
      use_ln=FLAGS.use_ln,
      nb_triplet_fts=FLAGS.nb_triplet_fts,
      nb_heads=FLAGS.nb_heads,
      branching_structure=branching_structure
  )
  checkpoint_path = os.path.join(FLAGS.checkpoint_path, FLAGS.load_checkpoint_path)
  if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
  model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      encoder_init=FLAGS.encoder_init,
      use_lstm=FLAGS.use_lstm,
      learning_rate=FLAGS.learning_rate,
      grad_clip_max_norm=FLAGS.grad_clip_max_norm,
      checkpoint_path=checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dropout_prob=FLAGS.dropout_prob,
      hint_teacher_forcing=FLAGS.hint_teacher_forcing,
      hint_repred_mode=FLAGS.hint_repred_mode,
      nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
      num_layers=FLAGS.num_layers,
      activation=FLAGS.activation,
      attention_dropout=FLAGS.attention_dropout_prob,
      norm_first_att=FLAGS.norm_first_att,
      node_readout=FLAGS.node_readout,
      use_projection=FLAGS.use_projection,
      projection_dim=FLAGS.projection_dim,
  )
  if not FLAGS.use_graph_fts:
      is_graph_fts_avail = [False] * len(is_graph_fts_avail)

  eval_model = clrs.models.BaselineModel(
      spec=spec_list,
      dummy_trajectory=[next(t) for t in val_samplers],
      **model_params
  )
  if FLAGS.chunked_training:
    train_model = clrs.models.BaselineModelChunked(
        spec=spec_list,
        dummy_trajectory=[next(t) for t in train_samplers],
        **model_params
        )
  else:
    train_model = eval_model
  length_idx = 0

  for algo_idx in range(len(train_samplers)):
    logging.info(f"{FLAGS.algorithms[algo_idx]} has graph features: {is_graph_fts_avail[algo_idx]}")
    
  def solve_logistic_regression(gradients, labels):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0).fit(gradients, labels)
    logging.info('Logistic regression score: %s', clf.score(gradients, labels))
    return clf.coef_
    
  def l2_norm(params):
    return np.linalg.norm(params)

  relative_errors = []
  for run in range(FLAGS.runs):
    step = 0
    while step < FLAGS.train_steps:
      feedback_list = [next(t) for t in train_samplers]

      # Initialize model.
      if step == 0:
        all_features = [f.features for f in feedback_list]
        train_model.init(all_features, is_graph_fts_avail, FLAGS.seed + 1)

        # Initialize projection matrix
        gradient_dim = 0
        for key in train_model.params:
            for param in train_model.params[key]:
              if 'processor' in key and "layer_norm" not in key:
                gradient_dim += np.prod(train_model.params[key][param].shape)

        # collect gradients
        gradients_dir = f"./gradients/processor_{FLAGS.processor_type}_layers_{FLAGS.num_layers}_dim_{FLAGS.hidden_size}_" \
                      + "seed_{}_projection_dim_{}_".format(FLAGS.gradient_projection_seed, FLAGS.projection_dim) \
                      + "_".join([algorithm[:3] for algorithm in FLAGS.algorithms])
        if not os.path.exists(gradients_dir):
          os.makedirs(gradients_dir)
                    
        np.random.seed(FLAGS.gradient_projection_seed); project_dim = FLAGS.gradient_projection_dim
        matrix_P = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(np.float32)
        matrix_P *= 1 / np.sqrt(project_dim)

        # load gradients & solve logistic regression
        gradients, labels = [], []; count = 0
        for file in os.listdir(gradients_dir):
          if "gradients" in file:
            gradients.append(np.load(os.path.join(gradients_dir, file)))
            count += 1
            if count >= 20: # TODO: Need to be tuned (how many samples are used for linear regression)
              break
        count = 0
        for file in os.listdir(gradients_dir):
          if "labels" in file:
            labels.append(np.load(os.path.join(gradients_dir, file)))
            count += 1
            if count >= 20:
              break
        gradients = np.concatenate(gradients, axis=0)
        labels = np.concatenate(labels, axis=0)
        # min_length = min(gradients.shape[0], labels.shape[0])
        # gradients = gradients[:min_length]; labels = labels[:min_length]
        # logging.info('Gradients shape: %s', gradients.shape)

        coef = solve_logistic_regression(gradients, labels)
        logging.info('Logistic regression weights: %s', l2_norm(coef.flatten()))
        update_weight = matrix_P @ coef.T.flatten() # restore weight dimension

        # rescale the update weight w.r.t. pretrained weights
        utils.restore_model(train_model, 'best.pkl', change_algo_index=FLAGS.change_algo_index) # load pretrained model
        logging.info('Pretrained model weights: %s', l2_norm(utils.get_pretrained_model_weights(train_model, FLAGS.layer)))
        
        update_weight = (update_weight / l2_norm(update_weight)) * (FLAGS.perturb_ratio * l2_norm(utils.get_pretrained_model_weights(train_model, FLAGS.layer)))
        logging.info('Update weight: %s', l2_norm(update_weight))

      # Training step. Note: if there are multiple samplers, it will apply one step per sampler
      for algo_idx in range(len(train_samplers)):
        feedback = feedback_list[algo_idx]
        rng_key, new_rng_key = jax.random.split(rng_key)
        if FLAGS.chunked_training:
          # In chunked training, we must indicate which training length we are
          # using, so the model uses the correct state.
          length_and_algo_idx = (length_idx, algo_idx)
        else:
          # In non-chunked training, all training lengths can be treated equally,
          # since there is no state to maintain between batches.
          length_and_algo_idx = algo_idx
        
        # first compute pretrained outputs
        utils.assign_to_model_parameters(train_model)
        outputs_list_1 = train_model.compute_output_function_values(rng_key, feedback, is_graph_fts_avail[algo_idx], length_and_algo_idx)

        # then compute outputs with perturbed weights
        utils.assign_to_model_parameters(train_model, update_weight, FLAGS.layer)
        outputs_list_2 = train_model.compute_output_function_values(rng_key, feedback, is_graph_fts_avail[algo_idx], length_and_algo_idx)
        
        for idx, _ in enumerate(outputs_list_1):
          error = l2_norm(outputs_list_1[idx] - outputs_list_2[idx])
          relative_errors.append(np.square(error / l2_norm(outputs_list_1[idx])))
      step += 1
    logging.info("Relative errors: {} Stds: {}".format(np.mean(relative_errors), np.std(relative_errors)))

  logging.info("Relative errors: {} Stds: {}".format(np.mean(relative_errors), np.std(relative_errors)))
  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)

