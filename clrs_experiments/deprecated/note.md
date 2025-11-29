```
conda create -n clrs python=3.10

pip install -e .

Feedback(
    features=Features(
    inputs=(DataPoint(name="pos",        location=node,  type=scalar,    data=Array(4, 16)), DataPoint(name="s", location=node,  type=mask_one,  data=Array(4, 16)), DataPoint(name="A", location=edge,  type=scalar,    data=Array(4, 16, 16)), DataPoint(name="adj",       location=edge,  type=mask,      data=Array(4, 16, 16))), 
    hints=(DataPoint(name="reach_h",       location=node,  type=mask,      data=Array(10, 4, 16)), DataPoint(name="pi_h",  location=node,  type=pointer,   data=Array(10, 4, 16))), lengths=array([3., 1., 3., 3.])), 
    
    outputs=[DataPoint(name="pi",    location=node,  type=pointer,   data=Array(4, 16))])

```

### Structure of model params (MPNN)

```
net/mpnn_aggr_clrs_processor/layer_norm offset (192,)
net/mpnn_aggr_clrs_processor/layer_norm scale (192,)
net/mpnn_aggr_clrs_processor/linear b (192,)
net/mpnn_aggr_clrs_processor/linear w (384, 192)
net/mpnn_aggr_clrs_processor/linear_1 b (192,)
net/mpnn_aggr_clrs_processor/linear_1 w (384, 192)
net/mpnn_aggr_clrs_processor/linear_2 b (192,)
net/mpnn_aggr_clrs_processor/linear_2 w (192, 192)
net/mpnn_aggr_clrs_processor/linear_3 b (192,)
net/mpnn_aggr_clrs_processor/linear_3 w (192, 192)
net/mpnn_aggr_clrs_processor/linear_4 b (192,)
net/mpnn_aggr_clrs_processor/linear_4 w (384, 192)
net/mpnn_aggr_clrs_processor/linear_5 b (192,)
net/mpnn_aggr_clrs_processor/linear_5 w (192, 192)
net/mpnn_aggr_clrs_processor/mlp/~/linear_0 b (192,)
net/mpnn_aggr_clrs_processor/mlp/~/linear_0 w (192, 192)
net/mpnn_aggr_clrs_processor/mlp/~/linear_1 b (192,)
net/mpnn_aggr_clrs_processor/mlp/~/linear_1 w (192, 192)
net/~_construct_encoders_decoders/algo_0_A_enc_linear b (192,)
net/~_construct_encoders_decoders/algo_0_A_enc_linear w (1, 192)
net/~_construct_encoders_decoders/algo_0_adj_enc_linear b (192,)
net/~_construct_encoders_decoders/algo_0_adj_enc_linear w (1, 192)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear b (16,)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear w (16, 16)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_1 b (16,)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_1 w (16, 16)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_2 b (16,)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_2 w (16, 16)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_3 b (1,)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_3 w (16, 1)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear b (16,)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear w (16, 16)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_1 b (16,)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_1 w (16, 16)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_2 b (16,)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_2 w (16, 16)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_3 b (1,)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_3 w (16, 1)
net/~_construct_encoders_decoders/algo_0_pi_h_enc_linear b (192,)
net/~_construct_encoders_decoders/algo_0_pi_h_enc_linear w (1, 192)
net/~_construct_encoders_decoders/algo_0_pos_enc_linear b (192,)
net/~_construct_encoders_decoders/algo_0_pos_enc_linear w (1, 192)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear b (16,)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear w (576, 16)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear_1 b (16,)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear_1 w (192, 16)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear_2 b (16,)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear_2 w (192, 16)
net/~_construct_encoders_decoders/algo_0_reach_h_dec_linear b (1,)
net/~_construct_encoders_decoders/algo_0_reach_h_dec_linear w (16, 1)
net/~_construct_encoders_decoders/algo_0_reach_h_enc_linear b (192,)
net/~_construct_encoders_decoders/algo_0_reach_h_enc_linear w (1, 192)
net/~_construct_encoders_decoders/algo_0_s_enc_linear b (192,)
net/~_construct_encoders_decoders/algo_0_s_enc_linear w (1, 192)
Processor params:  407424 Encoder params:  2304 Decoder params:  1683
```

### Structure of model params (Edge Transformer)


```
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/EdgeAttention/linear w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/EdgeAttention/linear_1 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/EdgeAttention/linear_2 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/EdgeAttention/linear_3 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/EdgeAttention/linear_4 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/layer_norm scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/layer_norm_1 offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/layer_norm_1 scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/linear b (48,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/linear w (24, 48)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/linear_1 b (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/linear_1 w (48, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/layer_norm scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/EdgeAttention/linear w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/EdgeAttention/linear_1 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/EdgeAttention/linear_2 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/EdgeAttention/linear_3 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/EdgeAttention/linear_4 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/layer_norm scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/layer_norm_1 offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/layer_norm_1 scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/linear b (48,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/linear w (24, 48)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/linear_1 b (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/linear_1 w (48, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/layer_norm scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/EdgeAttention/linear w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/EdgeAttention/linear_1 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/EdgeAttention/linear_2 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/EdgeAttention/linear_3 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/EdgeAttention/linear_4 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/layer_norm scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/layer_norm_1 offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/layer_norm_1 scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/linear b (48,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/linear w (24, 48)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/linear_1 b (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/linear_1 w (48, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/layer_norm scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/EdgeAttention/linear w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/EdgeAttention/linear_1 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/EdgeAttention/linear_2 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/EdgeAttention/linear_3 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/EdgeAttention/linear_4 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/layer_norm scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/layer_norm_1 offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/layer_norm_1 scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/linear b (48,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/linear w (24, 48)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/linear_1 b (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/linear_1 w (48, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/layer_norm scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/EdgeAttention/linear w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/EdgeAttention/linear_1 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/EdgeAttention/linear_2 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/EdgeAttention/linear_3 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/EdgeAttention/linear_4 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/layer_norm scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/layer_norm_1 offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/layer_norm_1 scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/linear b (48,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/linear w (24, 48)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/linear_1 b (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/linear_1 w (48, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/layer_norm scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/EdgeAttention/linear w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/EdgeAttention/linear_1 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/EdgeAttention/linear_2 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/EdgeAttention/linear_3 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/EdgeAttention/linear_4 w (24, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/layer_norm scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/layer_norm_1 offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/layer_norm_1 scale (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/linear b (48,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/linear w (24, 48)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/linear_1 b (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/linear_1 w (48, 24)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/layer_norm offset (24,)
net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/layer_norm scale (24,)
net/edge_t_clrs_processor/linear_1 b (24,)
net/edge_t_clrs_processor/linear_1 w (144, 24)
net/~_construct_encoders_decoders/algo_0_A_enc_linear b (24,)
net/~_construct_encoders_decoders/algo_0_A_enc_linear w (1, 24)
net/~_construct_encoders_decoders/algo_0_adj_enc_linear b (24,)
net/~_construct_encoders_decoders/algo_0_adj_enc_linear w (1, 24)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear b (4,)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear w (4, 4)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_1 b (4,)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_1 w (4, 4)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_2 b (4,)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_2 w (4, 4)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_3 b (1,)
net/~_construct_encoders_decoders/algo_0_pi_dec_linear_3 w (4, 1)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear b (4,)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear w (4, 4)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_1 b (4,)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_1 w (4, 4)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_2 b (4,)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_2 w (4, 4)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_3 b (1,)
net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_3 w (4, 1)
net/~_construct_encoders_decoders/algo_0_pi_h_enc_linear b (24,)
net/~_construct_encoders_decoders/algo_0_pi_h_enc_linear w (1, 24)
net/~_construct_encoders_decoders/algo_0_pos_enc_linear b (24,)
net/~_construct_encoders_decoders/algo_0_pos_enc_linear w (1, 24)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear b (4,)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear w (72, 4)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear_1 b (4,)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear_1 w (48, 4)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear_2 b (4,)
net/~_construct_encoders_decoders/algo_0_projector_projector_linear_2 w (24, 4)
net/~_construct_encoders_decoders/algo_0_reach_h_dec_linear b (1,)
net/~_construct_encoders_decoders/algo_0_reach_h_dec_linear w (4, 1)
net/~_construct_encoders_decoders/algo_0_reach_h_enc_linear b (24,)
net/~_construct_encoders_decoders/algo_0_reach_h_enc_linear w (1, 24)
net/~_construct_encoders_decoders/algo_0_s_enc_linear b (24,)
net/~_construct_encoders_decoders/algo_0_s_enc_linear w (1, 24)
```

#### Gradients

```
'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/EdgeAttention/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/EdgeAttention/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/EdgeAttention/linear_2', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/EdgeAttention/linear_3', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/EdgeAttention/linear_4', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/layer_norm', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/layer_norm_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/FFN/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer/layer_norm', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/EdgeAttention/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/EdgeAttention/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/EdgeAttention/linear_2', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/EdgeAttention/linear_3', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/EdgeAttention/linear_4', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/layer_norm', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/layer_norm_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/FFN/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_1/layer_norm', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/EdgeAttention/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/EdgeAttention/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/EdgeAttention/linear_2', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/EdgeAttention/linear_3', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/EdgeAttention/linear_4', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/layer_norm', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/layer_norm_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/FFN/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_2/layer_norm', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/EdgeAttention/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/EdgeAttention/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/EdgeAttention/linear_2', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/EdgeAttention/linear_3', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/EdgeAttention/linear_4', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/layer_norm', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/layer_norm_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/FFN/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_3/layer_norm', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/EdgeAttention/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/EdgeAttention/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/EdgeAttention/linear_2', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/EdgeAttention/linear_3', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/EdgeAttention/linear_4', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/layer_norm', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/layer_norm_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/FFN/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_4/layer_norm', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/EdgeAttention/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/EdgeAttention/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/EdgeAttention/linear_2', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/EdgeAttention/linear_3', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/EdgeAttention/linear_4', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/layer_norm', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/layer_norm_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/linear', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/FFN/linear_1', 'net/edge_t_clrs_processor/EdgeTransformer/ET_Layer_5/layer_norm', 'net/edge_t_clrs_processor/linear_1', 'net/~_construct_encoders_decoders/algo_0_A_enc_linear', 'net/~_construct_encoders_decoders/algo_0_adj_enc_linear', 'net/~_construct_encoders_decoders/algo_0_i_dec_linear', 'net/~_construct_encoders_decoders/algo_0_i_enc_linear', 'net/~_construct_encoders_decoders/algo_0_j_dec_linear', 'net/~_construct_encoders_decoders/algo_0_j_enc_linear', 'net/~_construct_encoders_decoders/algo_0_key_enc_linear', 'net/~_construct_encoders_decoders/algo_0_pi_dec_linear', 'net/~_construct_encoders_decoders/algo_0_pi_dec_linear_1', 'net/~_construct_encoders_decoders/algo_0_pi_dec_linear_2', 'net/~_construct_encoders_decoders/algo_0_pi_dec_linear_3', 'net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear', 'net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_1', 'net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_2', 'net/~_construct_encoders_decoders/algo_0_pi_h_dec_linear_3', 'net/~_construct_encoders_decoders/algo_0_pi_h_enc_linear', 'net/~_construct_encoders_decoders/algo_0_pos_enc_linear', 'net/~_construct_encoders_decoders/algo_0_pred_dec_linear', 'net/~_construct_encoders_decoders/algo_0_pred_dec_linear_1', 'net/~_construct_encoders_decoders/algo_0_pred_dec_linear_2', 'net/~_construct_encoders_decoders/algo_0_pred_dec_linear_3', 'net/~_construct_encoders_decoders/algo_0_pred_h_dec_linear', 'net/~_construct_encoders_decoders/algo_0_pred_h_dec_linear_1', 'net/~_construct_encoders_decoders/algo_0_pred_h_dec_linear_2', 'net/~_construct_encoders_decoders/algo_0_pred_h_dec_linear_3', 'net/~_construct_encoders_decoders/algo_0_pred_h_enc_linear', 'net/~_construct_encoders_decoders/algo_0_pred_mask_dec_linear', 'net/~_construct_encoders_decoders/algo_0_projector_projector_linear', 'net/~_construct_encoders_decoders/algo_0_projector_projector_linear_1', 'net/~_construct_encoders_decoders/algo_0_projector_projector_linear_2', 'net/~_construct_encoders_decoders/algo_0_reach_h_dec_linear', 'net/~_construct_encoders_decoders/algo_0_reach_h_enc_linear', 'net/~_construct_encoders_decoders/algo_0_s_enc_linear'
```

### Haiku

Transforms a function using Haiku modules into a pair of pure functions.

For a function ``out = f(*a, **k)`` this function returns a pair of two pure
functions that call ``f(*a, **k)`` explicitly collecting and injecting
parameter values::

    params = init(rng, *a, **k)
    out = apply(params, rng, *a, **k)

Note that the ``rng`` argument is typically not required for ``apply`` and
passing ``None`` is accepted.

The first thing to do is to define a :class:`Module`. A module encapsulates
some parameters and a computation on those parameters:

>>> class MyModule(hk.Module):
...   def __call__(self, x):
...     w = hk.get_parameter("w", [], init=jnp.zeros)
...     return x + w

Next, define some function that creates and applies modules. We use
:func:`transform` to transform that function into a pair of functions that
allow us to lift all the parameters out of the function (``f.init``) and
apply the function with a given set of parameters (``f.apply``):

>>> def f(x):
...   a = MyModule()
...   b = MyModule()
...   return a(x) + b(x)
>>> f = hk.transform(f)

To get the initial state of the module call ``init`` with an example input:

>>> params = f.init(None, 1)
>>> params
{'my_module': {'w': ...Array(0., dtype=float32)},
'my_module_1': {'w': ...Array(0., dtype=float32)}}

You can then apply the function with the given parameters by calling
``apply`` (note that since we don't use Haiku's random number APIs to apply
our network we pass ``None`` as an RNG key):

>>> print(f.apply(params, None, 1))
2.0

It is expected that your program will at some point produce updated parameters
and you will want to re-apply ``apply``. You can do this by calling ``apply``
with different parameters:

>>> new_params = {"my_module": {"w": jnp.array(2.)},
...               "my_module_1": {"w": jnp.array(3.)}}
>>> print(f.apply(new_params, None, 2))
9.0

If your transformed function needs to maintain internal state (e.g. moving
averages in batch norm) then see :func:`transform_with_state`.

