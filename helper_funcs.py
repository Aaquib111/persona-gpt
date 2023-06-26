# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
import pandas as pd 
import einops
from tqdm.notebook import tqdm
import plotly.express as px
import webbrowser
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from functools import partial
from IPython.display import display, HTML
import circuitsvis as cv
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
import plotly.graph_objects as go

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


# !git clone https://github.com/callummcdougall/path_patching.git
import sys
sys.path.append('./path_patching/')  # replace <repo-name> with the name of the cloned repository
from path_patching import Node, IterNode, path_patch, act_patch

t.set_grad_enabled(False)

from plotly_utils import imshow, line, scatter, bar

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

# %%

from torch.utils.data import DataLoader, Dataset

class PromptCompletionDataset(Dataset):
    def __init__(self, model, prompt):
        self.model = model
        self.vocab_str = self.model.to_str_tokens(t.arange(self.model.cfg.d_vocab))
        self.prompt = prompt

    def __getitem__(self, idx):
        return self.prompt + self.vocab_str[idx]

    def __len__(self):
        return len(self.vocab_str)


def get_completion_sentiment(model: HookedTransformer, prompt: str, batch_size: int = 64) -> Float[Tensor, 'vocab']:
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # config = AutoConfig.from_pretrained(MODEL)
    classifier = AutoModelForSequenceClassification.from_pretrained(MODEL)

    sentiment = t.zeros(model.cfg.d_vocab).to(device)
    dataset = PromptCompletionDataset(model, prompt)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    for i, batch in tqdm(enumerate(dataloader)):
        class_output = classifier(**tokenizer(batch, return_tensors='pt', padding=True))
        class_scores = t.softmax(class_output.logits.squeeze(), dim=-1)
        
        if class_scores.dim() == 1:  # if only one item in the batch
            sentiment[i*batch_size] = class_scores[2] - class_scores[0]  # positive - negative
        else:
            sentiment[i*batch_size:(i+1)*batch_size] = class_scores[:, 2] - class_scores[:, 0]  # positive - negative
    
    return sentiment


def print_topk_predictions(model: HookedTransformer,
                           prompt: Optional[str] = None,
                           logits: Optional[Float[Tensor, 'seq vocab']] = None,
                           k=5) -> None:
    if prompt is not None:
        logits = model(prompt).squeeze(0)
    if logits.dim() == 3:
        logits = logits.squeeze(0) 
    top_probs, top_indices = t.topk(t.softmax(logits[-1], dim=-1), k=k)
    preds = "\t".join([f"{p.item():.3f}: '{model.to_string(idx)}'" 
                          for p, idx in zip(top_probs, top_indices)])
    print(preds)


def compute_completion_sentiment(logits: Float[Tensor, 'batch seq vocab'],
                                 vocab_sentiment: Float[Tensor, 'vocab'],
                                 prob_threshold: float = 0.01
                                 ) -> Float[Tensor, 'batch']:
    probs = t.softmax(logits[:, -1], dim=-1)
    probs[probs < prob_threshold] = 0
    sentiment = (probs * vocab_sentiment).sum(dim=-1) / probs.sum(dim=-1)
    if sentiment.dim() == 1:
        return sentiment.item()
    else:
        return sentiment.mean().item()


def plot_logit_attr(cache: ActivationCache,
                    pos_slice: int,
                    top_k: int = 10,
                    component_slice: slice = slice(None, None, 4),
                    add_b_U: bool = False,
                    show_plot: bool = True) -> None:
    resid_components, resid_labels = cache.accumulated_resid(return_labels=True,
                                                                    # incl_mid=True,
                                                                    apply_ln=True,
                                                                    pos_slice=pos_slice,)
    
    redux_resid_components, redux_resid_labels = resid_components[component_slice], resid_labels[component_slice]
    resid_attrs = redux_resid_components @ cache.model.W_U 
    if add_b_U:
        resid_attrs = resid_attrs + cache.model.b_U
    top_logits, top_tokens = resid_attrs.mean(1).topk(top_k, dim=-1) # mean gets rid of batch dim
    top_tokens_list = [cache.model.to_str_tokens(t) for t in top_tokens]

    fig = go.Figure(data=go.Heatmap(
        z=top_logits.cpu().numpy(),
        y=redux_resid_labels,
        text=top_tokens_list,
        texttemplate='%{text}',
    ))
    fig.show()

    return top_logits, top_tokens

# %%