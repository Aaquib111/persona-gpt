#%% 
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import time
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
import einops
from tqdm.notebook import tqdm
import plotly.express as px
import webbrowser
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
from rich import box
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP

# Make sure exercises are in the path
# chapter = r"chapter1_transformers"
# exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
# section_dir = (exercises_dir / "part3_indirect_object_identification").resolve()
# if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
sys.path.append('..')
# import part3_indirect_object_identification.tests as tests

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

MAIN = __name__ == "__main__"
# %%
def create_model():
    model = HookedTransformer.from_pretrained(
        'gpt2-small',
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True
    )
    model.set_use_attn_result(True)
    return model
if MAIN:
    model = create_model()
 
hook_filter = lambda name: name.endswith("ln1.hook_normalized") or name.endswith("attn.hook_result")
def get_3_caches(model, clean_input, corrupted_input, metric):
    # cache the activations and gradients of the clean inputs
    model.reset_hooks()
    clean_cache = {}
    def forward_cache_hook(act, hook):
        clean_cache[hook.name] = act.detach()
    model.add_hook(hook_filter, forward_cache_hook, "fwd")

    clean_grad_cache = {}
    def backward_cache_hook(act, hook):
        clean_grad_cache[hook.name] = act.detach()
    model.add_hook(hook_filter, backward_cache_hook, "bwd")

    value = metric(model(clean_input))
    value.backward()
    
    # cache the activations of the corrupted inputs
    model.reset_hooks()
    corrupted_cache = {}
    def forward_cache_hook(act, hook):
        corrupted_cache[hook.name] = act.detach()
    model.add_hook(hook_filter, forward_cache_hook, "fwd")
    model(corrupted_input)
    model.reset_hooks()
    
    clean_cache = ActivationCache(clean_cache, model)
    corrupted_cache = ActivationCache(corrupted_cache, model)
    clean_grad_cache = ActivationCache(clean_grad_cache, model)
    return clean_cache, corrupted_cache, clean_grad_cache
#%%

def split_layers_and_heads(act: Tensor, model: HookedTransformer) -> Tensor:
    return einops.rearrange(act, '(layer head) batch seq d_model -> layer head batch seq d_model',
                            layer=model.cfg.n_layers,
                            head=model.cfg.n_heads)

def acdc_nodes(model: HookedTransformer,
              clean_input: Tensor,
              corrupted_input: Tensor,
              metric: Callable[[Tensor], Tensor],
              threshold: float,
              create_model: Callable[[], HookedTransformer],
              attr_absolute_val: bool = False) -> Tuple[
                  HookedTransformer, Bool[Tensor, 'n_layer n_heads']]:
    '''
    Runs attribution-patching-based ACDC on the model, using the given metric and data.
    Returns the pruned model, and which heads were pruned.

    Arguments:
        model: the model to prune
        clean_input: the input to the model that contains should elicit the behavior we're looking for
        corrupted_input: the input to the model that should elicit random behavior
        metric: the metric to use to compare the model's performance on the clean and corrupted inputs
        threshold: the threshold below which to prune
        create_model: a function that returns a new model of the same type as the input model
        attr_absolute_val: whether to take the absolute value of the attribution before thresholding
    '''
    # get the 2 fwd and 1 bwd caches; cache "normalized" and "result" of attn layers
    clean_cache, corrupted_cache, clean_grad_cache = get_3_caches(model, clean_input, corrupted_input, metric)

    # take all pairs of heads, 
    # edges = [
    #       ((layer_sender, head_sender), (layer_receiver, head_receiver))
    #       for layer_sender, layer_receiver in itertools.product(range(model.cfg.n_layer), repeat=2)
    #       for head_sender, head_receiver in itertools.product(range(model.cfg.n_heads), repeat=2)
    #       if layer_sender < layer_receiver
    # ]

    # compute first-order Taylor approximation for each node to get the attribution
    clean_head_act = clean_cache.stack_head_results()
    corr_head_act = corrupted_cache.stack_head_results()
    clean_grad_act = clean_grad_cache.stack_head_results()

    # compute attributions of each node
    node_attr = (clean_head_act - corr_head_act) * clean_grad_act
    # separate layers and heads, sum over d_model (to complete the dot product), batch, and seq
    node_attr = split_layers_and_heads(node_attr, model).sum((2, 3, 4))

    if attr_absolute_val:
        node_attr = node_attr.abs()

    # prune all nodes whose attribution is below the threshold
    should_prune = node_attr < threshold
    pruned_model = create_model()
    for layer, head in itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)):
        if should_prune[layer, head]:
            # yeet!
            pruned_model.W_Q[layer, head].zero_()

    # compute metric on pruned subgraph vs whole graph
    # metric_full_graph = metric(model(clean_input))
    # metric_pruned_graph = metric(pruned_model(clean_input))

    # return the pruned subgraph, which heads were pruned, and the metrics
    return pruned_model, should_prune#, metric_full_graph, metric_pruned_graph
#%%
if MAIN:
    
    

    def logits_to_ave_logit_diff_2(logits: Float[Tensor, "batch seq d_vocab"], ioi_dataset: IOIDataset = ioi_dataset, per_prompt=False):
        '''
        Returns logit difference between the correct and incorrect answer.

        If per_prompt=True, return the array of differences rather than the average.
        '''

        # Only the final logits are relevant for the answer
        # Get the logits corresponding to the indirect object / subject tokens respectively
        io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
        s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
        # Find logit difference
        answer_logit_diff = io_logits - s_logits
        return answer_logit_diff if per_prompt else answer_logit_diff.mean()

    with t.no_grad():
        ioi_logits_original = model(ioi_dataset.toks)
        abc_logits_original = model(abc_dataset.toks)

    ioi_per_prompt_diff = logits_to_ave_logit_diff_2(ioi_logits_original, per_prompt=True)
    abc_per_prompt_diff = logits_to_ave_logit_diff_2(abc_logits_original, per_prompt=True)

    ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original).item()
    abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original).item()

    def ioi_metric(
        logits: Float[Tensor, "batch seq d_vocab"],
        clean_logit_diff: float = ioi_average_logit_diff,
        corrupted_logit_diff: float = abc_average_logit_diff,
        ioi_dataset: IOIDataset = ioi_dataset,
    ) -> float:
        '''
        We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset), 
        and -1 when performance has been destroyed (i.e. is same as ABC dataset).
        '''
        patched_logit_diff = logits_to_ave_logit_diff_2(logits, ioi_dataset)
        return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

#%%
def format_heads_pruned(should_prune: Bool[Tensor, 'n_layer n_heads']) -> str:
    should_prune = should_prune.cpu().numpy()
    return ", ".join([
        f"L{layer}H{head}"
        for layer, head in zip(*should_prune.nonzero())
    ])

#%%
def format_heads_pruned(should_prune: Bool[Tensor, 'n_layer n_heads']) -> str:
    should_prune = should_prune.cpu().numpy()
    return ", ".join([
        f"L{layer}H{head}"
        for layer, head in zip(*should_prune.nonzero())
    ])

# %%
if MAIN:
    st = time.time()
    pruned_model, should_prune = acdc_nodes(model, ioi_dataset.toks,
                                             abc_dataset.toks, ioi_metric,
                                             threshold=0.02,
                                             create_model=create_model,
                                             attr_absolute_val=True)
    print(f"Time taken: {time.time() - st:.2f}s")
    print(f"Number of heads pruned: {should_prune.sum()}, out of {should_prune.numel()}")
    print(f"Nodes in the circuit: {format_heads_pruned(should_prune.logical_not())}")
# %%
