""" 
HistoGPT Inference Helper Functions
Author: Manuel Tran / Helmholtz Munich
"""

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
from tqdm import tqdm
from typing import Dict, List, Union


def move_to(object: Union[torch.Tensor, Dict, List, None], device: torch.device):
    # move single tensor to device
    if torch.is_tensor(object):
        if object.dtype == torch.float64:
            object = object.float()
        return object.to(device)
    elif object is None:
        return None
    # move list of tensors to device
    elif isinstance(object, list):
        object = [move_to(v, device) for v in object]
        return object
    # move dict of tensors to device
    elif isinstance(object, dict):
        for key, value in object.items():
            # move single value tensor to device
            if isinstance(value, torch.Tensor):
                object[key] = move_to(value, device)
            # move list of value tensors to device
            elif isinstance(value, list):
                object[key] = [move_to(v, device) for v in value]
        return object
    else:
        raise TypeError("Invalid type for move_to")


def generate(
    model: nn.Module,
    prompt: torch.Tensor,
    inputs: List[torch.Tensor],  # [feats, coords]
    length: int = 256,
    end_token: int = 2,  # '</s>' = 2, '.' = 4, 'mm' = 518
    top_k: int = 40,
    top_p: float = 0.95,
    temp: float = 0.7,
    device: torch.device = 'cuda',
):
    """  
    autoregressive generation using temperature, top-k, and top-p sampling
    """
    feats = move_to(inputs[0], device)
    coords = move_to(inputs[1], device)
    out = prompt.to(device)
    model.eval()

    with torch.no_grad():
        for _ in tqdm(range(length), leave=False):
            text = out

            if device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(text, feats, coords).logits
            else:
                logits = model(text, feats, coords).logits

            if temp > 0:
                logits = logits[:, -1, :] / temp
                top_k_warper = TopKLogitsWarper(top_k=top_k)
                top_p_warper = TopPLogitsWarper(top_p=top_p)
                logits = top_k_warper(None, logits)
                logits = top_p_warper(None, logits)
                probs = F.softmax(logits, dim=-1)
                probs = probs.squeeze(0)
                pred = torch.multinomial(probs, num_samples=1)
            else:
                logits = logits[:, -1, :]
                pred = torch.argmax(logits, dim=1)

            if pred == end_token:
                break

            out = torch.cat((out, pred.unsqueeze(0)), dim=1)

    return out


def chat_gpt(client, prompt):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        #model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )
    return response.choices[0].message.content


def api_call(client, prompt, retries=10):
    for i in range(retries):
        try:
            response = chat_gpt(client, prompt)
            return response
        except Exception as E:
            wait_time = (2**i) + random.random()
            print(f"Error occurred: {E}. Retry #{i+1} in {wait_time} seconds.")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded.")
