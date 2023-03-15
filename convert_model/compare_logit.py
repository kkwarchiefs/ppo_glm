#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : compare_logit.py
# @Author: 罗锦文
# @Date  : 2023/3/15
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import tritonclient.http as httpclient
model_name = "REL_model_onnx"
device = torch.device('cuda:7')
model_path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/glm_0.5"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True, return_dict=False)
model = model.half().to(device).eval()
triton_client = httpclient.InferenceServerClient(url='10.212.207.33:8000', connection_timeout=300, network_timeout=300)

for query_text, response_text in [('什么人不能喝三七粉', '服用三七粉期间,孕妇和儿童不宜使用。 三七粉是处方药,不是药品。 过量服用会引起中毒。')]:
    temp_inputs = tokenizer(query_text + "[gMASK]", return_tensors="pt", padding=True)
    temp_inputs = tokenizer.build_inputs_for_generation(temp_inputs, targets=response_text, max_gen_length=512, padding=False).to(device)
    logit =  model(**temp_inputs)
    print('logit', logit.cpu().numpy())
    temp_inputs.to("cpu")
    inputs = []
    inputs.append(httpclient.InferInput('input_ids', list(temp_inputs['input_ids'].shape), 'INT64'))
    inputs.append(httpclient.InferInput('position_ids', list(temp_inputs['position_ids'].shape), 'INT64'))
    inputs.append(httpclient.InferInput('attention_mask', list(temp_inputs['attention_mask'].shape), 'INT64'))
    inputs[0].set_data_from_numpy(temp_inputs['input_ids'].numpy())
    inputs[1].set_data_from_numpy(temp_inputs['position_ids'].numpy())
    inputs[2].set_data_from_numpy(temp_inputs['attention_mask'].numpy())
    output = httpclient.InferRequestedOutput('output')
    results = triton_client.infer(
        "REL_model_onnx",
        inputs,
        model_version='1',
        outputs=[output],
        request_id='1',
        timeout=300 * 1000
    )
    results = results.as_numpy('output')
    print('results', results)
