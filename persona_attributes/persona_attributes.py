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
# %%
pos_tokens = model.to_tokens([' good', ' great', ' best', ' coolest', ' awesome', ' greatest', ' smart'])[:, 1]
neg_tokens = model.to_tokens([' bad', ' worst', ' worst', ' lamest', ' awful', ' horrible', ' dumb'])[:, 1]
def get_opinion(logits, pos_tokens=pos_tokens, neg_tokens=neg_tokens, per_prompt=False):
    final_logits = logits[:, -1, :]
    pos, neg = final_logits[:, pos_tokens].sum(), final_logits[:, neg_tokens].sum()
    #print(pos, neg)
    return (pos - neg).item() if per_prompt else (pos - neg).mean().item()

# REPLACE WITH DATASET
if MAIN:
    clean_str = [
        #'I am a Muslim, and I think religion is a',
        'I am a nerd, and I think computers are a'
    ]
    corrupt_str = [
        'I am a frog, and I think religion is a',
        'I am a goat, and I think computers are a'
    ]

    clean_toks = model.to_tokens(clean_str)
    corrupt_toks = model.to_tokens(corrupt_str)
    clean_logits, clean_cache = model.run_with_cache(clean_toks)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_toks)

    clean_opinion = get_opinion(
        clean_logits
    )
    corrupt_opinion = get_opinion(
        corrupt_logits
    )
def opinion_denoising_metric(logits, clean_opinion=clean_opinion, corrupt_opinion=corrupt_opinion):
    patched_opinion = get_opinion(logits)
    return (patched_opinion - corrupt_opinion) / (corrupt_opinion - clean_opinion)
def opinion_noising_metric(logits, clean_opinion=clean_opinion, corrupt_opinion=corrupt_opinion):
    patched_opinion = get_opinion(logits)
    return (patched_opinion - clean_opinion) / (corrupt_opinion - clean_opinion)

if MAIN:
    model.reset_hooks(including_permanent=True)
    results = act_patch(
        model=model,
        orig_input=clean_toks,#corrupt_toks,
        new_input=corrupt_toks,#clean_toks,
        patching_nodes=IterNode('z'),
        patching_metric=opinion_noising_metric,
        verbose=True
    )
    imshow(results['z'])
if MAIN:
    resid_result = act_patch(
        model=model,
        orig_input=clean_toks,#corrupt_toks,
        new_input=corrupt_toks,#clean_toks,
        patching_nodes=IterNode('resid_pre', seq_pos='each'),
        patching_metric=opinion_noising_metric,
        verbose=True,
    )
    imshow(resid_result['resid_pre'])
# %% ATTN PATTERNS
from IPython.display import display, HTML
import circuitsvis as cv
if MAIN:
	top_heads = [(15, 1), (17, 0), (21, 13), (20, 17), (28, 15)]
		# Get all their attention patterns
	attn_patterns_for_important_heads: Float[Tensor, "head q k"] = torch.stack([
		clean_cache["pattern", layer][:, head].mean(0)
		for layer, head in top_heads
	])

	# Display results
	display(HTML(f"<h2>Chosen Attribution Heads</h2>"))
	display(cv.attention.attention_patterns(
		attention = attn_patterns_for_important_heads,
		tokens = model.to_str_tokens(clean_toks[0]),
		attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
	))

# %%
