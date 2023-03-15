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
model_name = "REL_model_onnx"
device = torch.device('cuda:7')
model_path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/glm_0.5"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

for query_text, response_text in [('什么人不能喝三七粉', '服用三七粉期间,孕妇和儿童不宜使用。 三七粉是处方药,不是药品。 过量服用会引起中毒。')]:
    temp_inputs = tokenizer(query_text + "[gMASK]", return_tensors="pt", padding=True)
    temp_inputs = tokenizer.build_inputs_for_generation(temp_inputs, targets=response_text, max_gen_length=512, padding=False).to(device)
