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

#%% LOAD DATASET
import json
def load_data():
    with open("../persona_opinion.json") as f:
        data = json.load(f)
    pos_text = [d['sentence'] for d in data['positive']]
    neg_text = [d['sentence'] for d in data['negative']]
    unrelated_text = [d['sentence'] for d in data['unrelated']]
    return pos_text, neg_text, unrelated_text
def text_to_tokens(prompts: List[str]) -> Float[Tensor,"batch seq_len"]:
    return model.to_tokens(prompts, prepend_bos=False).to(device)


# %% SET UP DATA
from helper_funcs import get_completion_sentiment, compute_completion_sentiment

if MAIN:
    model.reset_hooks(including_permanent=True)
    pos_text, neg_text, unrelated_text = load_data()
    pos_toks = text_to_tokens(pos_text)
    neg_toks = text_to_tokens(neg_text)
    unrelated_toks = text_to_tokens(unrelated_text)

    pos_logits, pos_cache = model.run_with_cache(
        pos_toks
    )
    unrelated_logits, unrelated_cache = model.run_with_cache(
        unrelated_toks
    )

    vocab_sentiment = get_completion_sentiment(
        model,
        '',
        1
    )
    pos_sentiment = compute_completion_sentiment(
        pos_logits, 
        vocab_sentiment
    )
    unrelated_sentiment = compute_completion_sentiment(
        unrelated_logits,
        vocab_sentiment
    )  

# %% Sentiment Difference 
def sentiment_noising_metric(logits: Float[Tensor, 'batch seq vocab'],
                             completion_sentiment: Float[Tensor, 'vocab'],
                             clean_sentiment: float,
                             corrupt_sentiment: float):
    patched_sentiment = compute_completion_sentiment(logits, completion_sentiment)
    return (patched_sentiment - clean_sentiment) / (corrupt_sentiment - clean_sentiment)

def sentiment_denoising_metric(logits: Float[Tensor, 'batch seq vocab'],
                             completion_sentiment: Float[Tensor, 'vocab'],
                             clean_sentiment: float,
                             corrupt_sentiment: float):
    patched_sentiment = compute_completion_sentiment(logits, completion_sentiment)
    return (patched_sentiment - corrupt_sentiment) / (corrupt_sentiment - clean_sentiment)

if MAIN:
    model.reset_hooks(including_permanent=True)
    denoising_metric = partial(
        sentiment_denoising_metric,
        completion_sentiment=vocab_sentiment,
        clean_sentiment=pos_sentiment,
        corrupt_sentiment=unrelated_sentiment
    )
    results = act_patch(
        model=model,
        orig_input=unrelated_toks[:10],
        new_input=pos_toks[:10],
        patching_nodes=IterNode('z'),
        patching_metric=denoising_metric,
        verbose=True
    )
    imshow(results['z'])
# %%
