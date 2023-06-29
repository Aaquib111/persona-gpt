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
        'gpt2-xl',
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
    unrelated_text = [d['sentence'] for d in data['unrelated']]
    return pos_text, unrelated_text
def text_to_tokens(prompts: List[str]) -> Float[Tensor,"batch seq_len"]:
    return model.to_tokens(prompts).to(device)


# %% SET UP DATA
from helper_funcs import get_completion_sentiment, compute_completion_sentiment
most_common_str = [ ' the', ' be', ' to', ' of', ' and', \
                    ' a', ' in', ' that', ' have', ' I', \
                    ' it', ' for', ' not', ' on', ' with', \
                    ' he', ' as', ' you', ' do', ' at', ' this' ,\
                    ' but', ' his', ' by', ' from', ' they', ' we', \
                    ' say', ' her', ' she', ' or', ' an', ' will', \
                    ' my', ' one', ' all', ' would',' there',' their', \
                    ' what',' so',' up',' out',' if',' about',' who', \
                    ' get',' which',' go',' me']
most_common_toks = model.to_tokens(most_common_str, prepend_bos=False)[:, 0]

def print_topk_sents(model: HookedTransformer,
                           logits: Float[Tensor, 'seq vocab'],
                           most_common_toks: List[int],
                           vocab_sentiment,
                           k=5) -> None:
    logits[-1, most_common_toks] = -float('inf')
    probs = torch.softmax(logits[-1], dim=-1) * abs(vocab_sentiment)
    top_probs, top_indices = torch.topk(probs, k=k)
    preds = "\t".join([f"{p.item():.3f}: '{model.to_string(idx)}'" 
                          for p, idx in zip(top_probs, top_indices)])
    print(preds)

def compute_completion_sentiment(logits: Float[Tensor, 'batch seq vocab'],
                                 vocab_sentiment: Float[Tensor, 'vocab'],
                                 most_common_toks: List[int],
                                 k: int = 10
                                 ) -> Float[Tensor, 'batch']:
    logits[:, -1, most_common_toks] = -float('inf')
    probs = torch.softmax(logits[:, -1], dim=-1)
    probs *= vocab_sentiment
    sentiment = probs.sum(dim=-1)
    #print(sentiment)
    if sentiment.shape[0] == 1:
        return sentiment.item()
    else:
        return sentiment.mean().item()

def pos_indifferent_diff(
        logits: Float[Tensor, 'batch seq vocab'],
        positive_tokens,
        #indifferent_tokens,
        most_common_toks,
        per_prompt=False
        ):
    logits[:, -1, most_common_toks] = -float('inf')
    final_logits = logits[:, -1, :]
    pos_l = final_logits[:, positive_tokens]
    #unr_l = final_logits[:, indifferent_tokens]
    unr_l = 0

    return (pos_l - unr_l) if per_prompt else (pos_l - unr_l).mean()

#%%
if MAIN:
    model.reset_hooks(including_permanent=True)
    pos_text, unrelated_text = load_data()
    pos_toks = text_to_tokens(pos_text)[:, :-1] # For some reason, adds extra BOS token to only this
    unrelated_toks = text_to_tokens(unrelated_text)
    assert pos_toks.shape[-1] == unrelated_toks.shape[-1] and pos_toks[0, -1] != 50256 and unrelated_toks[0, -1] != 50256
    pos_logits, pos_cache = model.run_with_cache(
        pos_toks
    )
    unrelated_logits, unrelated_cache = model.run_with_cache(
        unrelated_toks
    )

    # vocab_sentiment = get_completion_sentiment(
    #     model,
    #     prompt='',
    #     batch_size=256
    # )
    # vocab_sentiment -= vocab_sentiment.mean()
    positive_tokens = model.to_tokens([' good', ' amazing', ' awesome', ' great', ' beautiful', ' pretty'])[:, 1]
    pos_sentiment = pos_indifferent_diff(
        pos_logits, 
        positive_tokens,
        most_common_toks
    )
    unrelated_sentiment = pos_indifferent_diff(
        unrelated_logits,
        positive_tokens,
        most_common_toks
    )  

print(pos_sentiment, unrelated_sentiment)

# %% Sentiment Difference 
def sentiment_noising_metric(logits: Float[Tensor, 'batch seq vocab'],
                             completion_sentiment: Float[Tensor, 'vocab'],
                             clean_sentiment: float,
                             corrupt_sentiment: float):
    patched_sentiment = compute_completion_sentiment(logits, completion_sentiment, most_common_toks)
    return (patched_sentiment - clean_sentiment) / (corrupt_sentiment - clean_sentiment)

def sentiment_denoising_metric(logits: Float[Tensor, 'batch seq vocab'],
                             completion_sentiment: Float[Tensor, 'vocab'],
                             clean_sentiment: float,
                             corrupt_sentiment: float):
    patched_sentiment = compute_completion_sentiment(logits, completion_sentiment, most_common_toks)
    return (patched_sentiment - corrupt_sentiment) / (corrupt_sentiment - clean_sentiment)

#%%
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
        orig_input=unrelated_toks,
        new_input=pos_toks,
        patching_nodes=IterNode('z'),
        patching_metric=denoising_metric,
        verbose=True
    )
    imshow(results['z'])
# %%
if MAIN:
    resid_result = act_patch(
        model=model,
        orig_input=unrelated_toks,
        new_input=pos_toks,
        patching_nodes=IterNode('resid_pre', seq_pos='each'),
        patching_metric=denoising_metric,
        verbose=True,
    )
    imshow(resid_result['resid_pre'])
# %% ATTN PATTERNS
from IPython.display import display, HTML
import circuitsvis as cv
if MAIN:
	top_heads = [(15, 1), (17, 0), (21, 13), (20, 17)]
		# Get all their attention patterns
	attn_patterns_for_important_heads: Float[Tensor, "head q k"] = torch.stack([
		pos_cache["pattern", layer][:, head].mean(0)
		for layer, head in top_heads
	])

	# Display results
	display(HTML(f"<h2>Chosen Attribution Heads</h2>"))
	display(cv.attention.attention_patterns(
		attention = attn_patterns_for_important_heads,
		tokens = model.to_str_tokens(pos_toks[0]),
		attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
	))

# %%
