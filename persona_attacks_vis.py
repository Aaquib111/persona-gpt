#%% IMPORTS
import os
import sys

import pandas
import numpy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import einops
from tqdm import tqdm

import transformer_lens
from transformer_lens import HookedTransformer

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


from accelerate import Accelerator
MAIN = __name__ == "__main__"

#%%
if MAIN:
    accelerator = Accelerator()
    device = accelerator.device

    #tokenizer = LlamaTokenizer.from_pretrained("7b")
    #hf_model = LlamaForCausalLM.from_pretrained("7b").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(device)
    #tokenizer = AutoTokenizer.from_pretrained("TheBloke/vicuna-7B-1.1-HF")
    #hf_model = AutoModelForCausalLM.from_pretrained("TheBloke/vicuna-7B-1.1-HF").to(device)
    #hf_model = accelerator.prepare(hf_model)
    #model = HookedTransformer.from_pretrained("llama-7b", hf_model=hf_model, device=device)

# %%

def converse(prompt, max_new_tokens=50, temperature=0.9, prepend_bos=True):
    # return model.generate(
    #     prompt,
    #     max_new_tokens=max_new_tokens,
    #     temperature=temperature,
    #     prepend_bos=prepend_bos
    # )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = hf_model.generate(
        input_ids, 
        do_sample=False, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_view(prompt, pos_token="good", neg_token="bad"):
    '''
        Method to return the logit score for the "positive" and "negative tokens"
    '''
    # Take positive token with and without prepended space
    pos_ids = torch.tensor([tokenizer(pos_token).input_ids, tokenizer(' ' + pos_token).input_ids])
    neg_ids = torch.tensor([tokenizer(neg_token).input_ids, tokenizer(' ' + neg_token).input_ids])
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    output = torch.softmax(hf_model(input_ids).logits[0, -1], dim=0) # Get last sequence position
    pos_logits_sum = output[pos_ids].sum()
    neg_logits_sum = output[neg_ids].sum()

    print(output.topk(10))
    return pos_logits_sum, neg_logits_sum
# %%
if MAIN:
    subjects = [
        "Sarah",
        "Jane",
        "Isabel",
        "Janice",
        "Alexandra",
        "Sam",
        "Michael",
        "Robert",
        "James"
    ]

    #from datasets import load_dataset
    #dataset = load_dataset("allenai/real-toxicity-prompts")
    import json

    with open('john_mary_data.json') as f:
        data = json.load(f)

    toxic = [d['Negative'] for d in data]
    nice = [d['Positive'] for d in data]
# %%
import numpy as np
def generate_dialog(num_interacts=5):
    pos_speaker, neg_speaker = np.random.choice(subjects, size=2, replace=False)
    dialog = ''
    for _ in range(num_interacts):
        # Positive response
        sentence = np.random.choice(nice)
        dialog += f'{pos_speaker}: {sentence} \n'

        # Toxic response
        sentence = np.random.choice(toxic)
        dialog += f'{neg_speaker}: {sentence} \n'

    return dialog, pos_speaker, neg_speaker
def generate_prompt(num_interacts=5, append_positive=False):
    dialog, pos, neg = generate_dialog(num_interacts=num_interacts)
    if append_positive:
        return dialog + f'{pos}: '
    return dialog + f'{neg}: '
# %%
if MAIN:
    for _ in range(5):
        print(converse(generate_prompt()))
        print('-------')