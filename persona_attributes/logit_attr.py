
#%% IMPORTS
from cgitb import Hook
from functools import partial
import os
import sys
from typing import List
from xml.etree.ElementInclude import include

import pandas
import numpy
import matplotlib.pyplot as plt
from regex import P

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import einops
from tqdm import tqdm

import transformer_lens
from transformer_lens import HookedTransformer, HookedEncoder, patching

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM


from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union

from accelerate import Accelerator
MAIN = __name__ == "__main__"
sys.path.append('..')
sys.path.append('../path_patching/')

from path_patching import Node, IterNode, path_patch, act_patch
from plotly_utils import imshow, line, scatter, bar

#%% LOAD MODEL
if MAIN:
    #accelerator = Accelerator()
    #device = accelerator.device
    device="cuda"

    model = HookedTransformer.from_pretrained(
        'gpt2-large',
		center_unembed=True,
		center_writing_weights=True,
		fold_ln=True,
		refactor_factored_attn_matrices=True,
    ).to(device)
    model.set_use_attn_result(True)
# %% TRANSLATING THROUGH SIGMOID
pos_tokens = model.to_tokens([' good', ' great', ' best', ' coolest', ' awesome', ' greatest', ' smart'])[:, 1]
neg_tokens = model.to_tokens([' bad', ' worst', ' worst', ' lamest', ' awful', ' horrible', ' dumb'])[:, 1]
def get_opinion_diff(logits, pos_tokens=pos_tokens, neg_tokens=neg_tokens, per_prompt=False):
    final_logits = logits[:, -1, :]
    pos, neg = final_logits[:, pos_tokens].sum(), final_logits[:, neg_tokens].sum()
    print(pos, neg)
    diff = (pos - neg) if per_prompt else (pos - neg).mean()
    return torch.tanh(torch.abs(diff)) # Return 0 for indifferent, higher for opinionated

#%% TRANSLATING THROUGH LINEAR
def get_post_final_ln_dir(model: HookedTransformer) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
    '''
    # SOLUTION
    return (model.W_U[:, pos_tokens] - model.W_U[:, neg_tokens]).sum(dim=-1)

#%% TRANSLATE THROUGH LAYERNORM
def get_activations(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    names: Union[str, List[str]]
):
    '''
    Uses hooks to return activations from the model.

    If names is a string, returns the activations for that hook name.
    If names is a list of strings, returns a dictionary mapping hook names to tensors of activations.
    '''
    names_list = [names] if isinstance(names, str) else names
    _, cache = model.run_with_cache(
        toks,
        return_type=None,
        names_filter=lambda name: name in names_list,
    )

    return cache[names] if isinstance(names, str) else cache

## FIT THE LINEAR REGRESSION
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

def LN_hook_names(layernorm: LayerNorm) -> Tuple[str, str]:
    '''
    Returns the names of the hooks immediately before and after a given layernorm.
    e.g. LN_hook_names(model.ln_final) returns ["blocks.2.hook_resid_post", "ln_final.hook_normalized"]
    '''
    if layernorm.name == "ln_final":
        input_hook_name = utils.get_act_name("resid_post", 35)
        output_hook_name = "ln_final.hook_normalized"
    else:
        layer, ln = layernorm.name.split(".")[1:]
        input_hook_name = utils.get_act_name("resid_pre" if ln=="ln1" else "resid_mid", layer)
        output_hook_name = utils.get_act_name('normalized', layer, ln)

    return input_hook_name, output_hook_name


pre_final_ln_name, post_final_ln_name = LN_hook_names(model.ln_final)
print(pre_final_ln_name, post_final_ln_name)

#%%
from sklearn.linear_model import LinearRegression
def get_ln_fit(
    model: HookedTransformer, data, layernorm: LayerNorm, seq_pos: Optional[int] = None
) -> Tuple[LinearRegression, float]:
    '''
    if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and the r^2 of the fit
    '''
    input_hook_name, output_hook_name = LN_hook_names(layernorm)
    # SOLUTION

    activations_dict = get_activations(model, data, [input_hook_name, output_hook_name])
    inputs = utils.to_numpy(activations_dict[input_hook_name].detach().cpu())
    outputs = utils.to_numpy(activations_dict[output_hook_name].detach().cpu())
    if seq_pos is None:
        inputs = einops.rearrange(inputs, "batch seq d_model -> (batch seq) d_model")
        outputs = einops.rearrange(outputs, "batch seq d_model -> (batch seq) d_model")
    elif isinstance(seq_pos, torch.Tensor):
        inputs = inputs[:, seq_pos.detach().cpu().numpy(), :]
        outputs = outputs[:, seq_pos.detach().cpu().numpy(), :]
        inputs = einops.rearrange(inputs, "batch seq d_model -> (batch seq) d_model")
        outputs = einops.rearrange(outputs, "batch seq d_model -> (batch seq) d_model")
    else:
        inputs = inputs[:, seq_pos, :]
        outputs = outputs[:, seq_pos, :]

    final_ln_fit = LinearRegression().fit(inputs, outputs)
    r2 = final_ln_fit.score(inputs, outputs)
    return (final_ln_fit, r2)

#%%
def get_pre_final_ln_dir(model: HookedTransformer, data) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in residual stream (pre ln_final, at sequence position 0) which
    most points in the direction of making an unbalanced classification.
    '''
    # SOLUTION
    post_final_ln_dir = get_post_final_ln_dir(model)

    final_ln_fit = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=4)[0]
    final_ln_coefs = torch.from_numpy(final_ln_fit.coef_).to(device)

    return final_ln_coefs.T @ post_final_ln_dir

#%%
def get_out_by_components(model: HookedTransformer, data) -> Float[Tensor, "component batch seq_pos emb"]:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2].
    The embeddings are the sum of token and positional embeddings.
    '''
    # SOLUTION
    embedding_hook_names = ["hook_embed", "hook_pos_embed"]
    head_hook_names = [utils.get_act_name("result", layer) for layer in range(model.cfg.n_layers)]
    mlp_hook_names = [utils.get_act_name("mlp_out", layer) for layer in range(model.cfg.n_layers)]

    all_hook_names = embedding_hook_names + head_hook_names + mlp_hook_names
    activations = get_activations(model, data, all_hook_names)
    out = (activations["hook_embed"] + activations["hook_pos_embed"]).unsqueeze(0)

    for head_hook_name, mlp_hook_name in zip(head_hook_names, mlp_hook_names):
        out = torch.concat([
            out, 
            einops.rearrange(activations[head_hook_name], "batch seq heads emb -> heads batch seq emb"),
            activations[mlp_hook_name].unsqueeze(0)
        ])

    return out
#%%
def indx_to_component_name(model):
    num_heads = model.cfg.n_heads
    num_layers = model.cfg.n_layers
    names = ['embed']
    for layer in range(num_layers):
        for head in range(num_heads):
            names.append(f'head {head}.{layer}')
        names.append(f'mlp {layer}')
    return names
indx_to_name = indx_to_component_name(model)
#%%
if MAIN:
    clean_str = [
        'I am a Christian, and I think religion is a',
        'I am an atheist, and I think atheism is a',
        #'I am a nerd, and I think computers are a'
    ]
    corrupt_str = [
        'I am an atheist, and I think religion is a',
        'I am a Christian, and I think atheism is a'
    ]
    unrelated_str = [
        'I am a crow, and I think religion is a',
        'I am a goat, and I think atheism is a'
    ]
    clean_toks = model.to_tokens(clean_str)
    corrupt_toks = model.to_tokens(corrupt_str)
    unrelated_toks = model.to_tokens(unrelated_str)
    data = torch.concat([clean_toks, corrupt_toks, unrelated_toks], dim=0)

    logits, cache = model.run_with_cache(data[0])

    clean_resid_dir = model.tokens_to_residual_directions(pos_tokens)#.sum(dim=1)
    corrupt_resid_dir = model.tokens_to_residual_directions(neg_tokens)#.sum(dim=1)
    logit_diff_dir = clean_resid_dir - corrupt_resid_dir


    out_by_components_seq4: Float[Tensor, "comp batch d_model"] = get_out_by_components(model, data)[:, :, 4, :]
    pre_final_ln_dir: Float[Tensor, "d_model"] = get_pre_final_ln_dir(model, data)
    # Get the size of the contributions for each component
    out_by_component_in_positive_dir = einops.einsum(
        out_by_components_seq4,
        pre_final_ln_dir,
        "comp batch d_model, d_model -> comp batch"
    )
    # Subtract the mean
    out_by_component_in_positive_dir -= out_by_component_in_positive_dir[:, 4:].mean(dim=1).unsqueeze(1)

    top_indices = torch.topk(out_by_component_in_positive_dir.sum(dim=1), k=50).indices
    for indx in top_indices:
        print(indx_to_name[indx])
# %% ATTN PATTERNS
from IPython.display import display, HTML
import circuitsvis as cv
if MAIN:
	top_heads = [(0, 14), (24, 19), (34, 4), (0, 3)]
		# Get all their attention patterns
	attn_patterns_for_important_heads: Float[Tensor, "head q k"] = torch.stack([
		cache["pattern", layer][:, head].mean(0)
		for layer, head in top_heads
	])

	# Display results
	display(HTML(f"<h2>Chosen Attribution Heads</h2>"))
	display(cv.attention.attention_patterns(
		attention = attn_patterns_for_important_heads,
		tokens = model.to_str_tokens(data[0]),
		attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
	))

# %% Changing residual vector at sequence position 4 before layer 24
from helper_funcs import plot_logit_attr
DATA = 3
logits, cache = model.run_with_cache(data[DATA])
# Before
plot_logit_attr(
    cache,
    pos_slice=4,
)
plot_logit_attr(
    cache,
    pos_slice=8,
)
plot_logit_attr(
    cache,
    pos_slice=-1,
)
# Adding vector
LAYER = 20
def get_resid_pre(prompt, layer):
    name = utils.get_act_name('resid_pre', layer)
    _, cache = model.run_with_cache(
        model.to_tokens(prompt),
        names_filter=lambda n: n == name
    )
    return cache[name][:, -1, :].squeeze() # Ignore the BOS token

act_add = get_resid_pre("Intense love for religion", LAYER)
act_sub = get_resid_pre('Incredible hatred towards God and religion', LAYER)
act_diff = act_add - act_sub

def steering_hook(activation, hook, seq_pos=4, coeff=1000, act_diff=act_diff):
    print('Patching')
    activation[:, seq_pos, :] += coeff * act_diff
    return activation

model.reset_hooks()
with model.hooks(fwd_hooks=[(utils.get_act_name('resid_pre', LAYER), steering_hook)]):
    logits, cache = model.run_with_cache(data[DATA])

# After
plot_logit_attr(
    cache,
    pos_slice=4,
)
plot_logit_attr(
    cache,
    pos_slice=8,
)
plot_logit_attr(
    cache,
    pos_slice=-1,
)
# %%
