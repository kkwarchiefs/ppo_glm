#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : compare_logit.py
# @Author: 罗锦文
# @Date  : 2023/3/15
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import codecs
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import tritonclient.http as httpclient
model_name = "REL_model_onnx"
device = torch.device('cuda:7')
model_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/reward_model_glm_10b_bak/final"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
triton_client = httpclient.InferenceServerClient(url='10.212.207.33:8000', connection_timeout=300, network_timeout=300)

pairlist = []
for line in open(sys.argv[1]):
    group = line.strip().split('\t')
    prompt = group[0]
    resp = eval(group[1])[0]
    label = int(group[2])
    pairlist.append((prompt, resp))
    # random.shuffle(pairlist)

for query_text, response_text in pairlist[:100]:
    prompt = query_text.replace("<|startofpiece|>", "").replace("[回答]", "").replace("[CLS]", "").replace("\n", "").replace(
        "<n>", "").replace("<|endoftext|>", "").replace("[gMASK]", "")
    response = response_text.replace("<|startofpiece|>", "").replace("<|endofpiece|>", "").replace("<|endoftext|>",
                                                                                              "").replace(" ", "")
    new_prompt = prompt + "[UNUSED1]" + response
    RM_input = tokenizer(new_prompt, max_length=512, padding=True)
    result = [torch.tensor([RM_input["input_ids"]]).numpy(), torch.tensor([RM_input["attention_mask"]]).numpy()]
    inputs = []
    inputs.append(httpclient.InferInput('input_ids', list(result[0].shape), 'INT64'))
    inputs.append(httpclient.InferInput('attention_mask', list(result[1].shape), 'INT64'))
    inputs[0].set_data_from_numpy(result[0])
    inputs[1].set_data_from_numpy(result[1])
    output = httpclient.InferRequestedOutput('output')
    # try:
    results = triton_client.infer(
        "RM_large_onnx",
        inputs,
        model_version='1',
        outputs=[output],
        request_id='1',
        timeout=300 * 1000
    )
    results = results.as_numpy('output')
    rewards = [torch.tensor(results[i][0]) for i in range(len(results))]
    print('mean_kl', query_text, (response_text,), rewards)
