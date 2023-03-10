import argparse
import os
from string import Template

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

model_name = "RM_model"
# device = torch.device('cuda:0')


RM_model_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/summarization_reward_model/checkpoint-58/"
RM_tokenizer = AutoTokenizer.from_pretrained(RM_model_path)
RM_model = AutoModelForSequenceClassification.from_pretrained(RM_model_path, num_labels=1, torchscript=True)
# RM_model.to(device)

input = RM_tokenizer("reward model's hash", return_tensors="pt")#.to(device)
print(input)
print(RM_model(input['input_ids'], input['attention_mask'], input['token_type_ids']))
traced_script_module = torch.jit.trace(RM_model, [input['input_ids'], input['attention_mask'], input['token_type_ids']])
jit_output = traced_script_module(input['input_ids'], input['attention_mask'], input['token_type_ids'])
print(jit_output)
os.makedirs(f"model_store/{model_name}/1", exist_ok=True)
traced_script_module.save(f"model_store/{model_name}/1/traced-model.pt")


