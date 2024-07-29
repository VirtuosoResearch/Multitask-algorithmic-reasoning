
def interpolate_models(model1_params, model2_params, alpha, include_keys=["attn", "mlp", "lm_head"]):
    interpolated_params = {}
    for k in model1_params.keys():
        if any([key in k for key in include_keys]):
            interpolated_params[k] = alpha * model1_params[k] + (1-alpha) * model2_params[k]
        else:
            interpolated_params[k] = model1_params[k] if alpha >= 0.5 else model2_params[k]
    return interpolated_params