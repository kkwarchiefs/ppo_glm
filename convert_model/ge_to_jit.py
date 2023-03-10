import argparse
import os
from string import Template

import torch

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from trl import AutoModelForSeq2SeqLMWithValueHead

model_name = "REL_model"
device = torch.device('cuda:7')


model_path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/cur_model/30630"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
inputs = tokenizer("reward model's hash"+"[gMASK]", return_tensors="pt")
ge_inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
temp_inputs = tokenizer.build_inputs_for_generation_from_tensor(inputs['input_ids'], inputs["input_ids"])
print(temp_inputs)
# temp_inputs.to(device)

model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torchscript=True, trust_remote_code=True, return_dict=False)
# model.set_tokenizer(tokenizer)
# model.to(device)
print(model(**temp_inputs))
# input = tokenizer("reward model's hash", return_tensors="pt").to(device)
traced_script_module = torch.jit.trace(model, [temp_inputs['input_ids'], temp_inputs['position_ids'], temp_inputs['attention_mask']])
jit_output = traced_script_module(temp_inputs['input_ids'], temp_inputs['position_ids'], temp_inputs['attention_mask'])
print(jit_output)
os.makedirs(f"model_store/{model_name}/1", exist_ok=True)
traced_script_module.save(f"model_store/{model_name}/1/traced-model.pt")


