#%% IMPORTS
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
from transformer_lens import HookedTransformer, patching

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    #tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    #hf_model = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device)
    model = HookedTransformer.from_pretrained(
        'gpt2-large',
		center_unembed=True,
		center_writing_weights=True,
		fold_ln=True,
		refactor_factored_attn_matrices=True,
    ).to(device)
# %%
from helper_funcs import plot_logit_attr

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
        'I am a Christian, and I think religion is a',
        #'I am a nerd, and I think computers are a'
        #'I am a woman, and I think that'
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
# %%
plot_logit_attr(
    clean_cache,
    pos_slice=-1,
)
# %%
plot_logit_attr(
    clean_cache,
    pos_slice=4,
)
#%% 
# Given, say fundamentalist in layer 20, what are the heads that 
# contribute the most to creating that token?


# %%
