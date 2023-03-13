# 需要安装以下环境，在线编码镜像 默认已经安装。
# pip3 install numpy
# pip3 install nvidia-pyindex
# pip3 install tritonclient[all]

import time

import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "REL_model_onnx"  # 模型目录名/venus注册模型名称
address = "10.212.204.89:8000"  # 机器地址
import json
rm_model_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/reward_model_glm_10b_bak/final"
tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
import torch

def GetRmBatchNumpy(prompt_list, response_list, RM_tokenizer):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    prompt_res = []
    for prompt, response in zip(prompt_list, response_list):
        prompt = prompt.replace("<|startofpiece|>", "").replace("[回答]", "").replace("[CLS]", "").replace("\n", "").replace("<n>", "")
        response = response.replace("<|startofpiece|>", "").replace("<|endofpiece|>", "").replace("<|endoftext|>", "").replace("<n>", "##402").replace(" ","")
        new_prompt = prompt + "[UNUSED1]" + response
        prompt_res.append(new_prompt[:500])
        # RM_input = RM_tokenizer((prompt + "[UNUSED1]" + response)[:300], max_length=512, padding=True)
        # input_ids_list.append(RM_input["input_ids"])
        # attention_mask_list.append(RM_input["attention_mask"])
        # token_type_ids_list.append(RM_input["token_type_ids"])
    RM_input = RM_tokenizer(prompt_res, max_length=512, padding=True)
    # print('RM_input:', RM_input)
    result = [torch.tensor(RM_input["input_ids"]).numpy(),  torch.tensor(RM_input["attention_mask"]).numpy()]
    # result = InputDict([("input_ids", torch.tensor(input_ids_list).to(cur_device)),("attention_mask", torch.tensor(attention_mask_list).to(cur_device)),("token_type_ids", torch.tensor(token_type_ids_list).to(cur_device))])
    return result

def request(count):
    triton_client = httpclient.InferenceServerClient(url=address)
    data_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/data/chatgpt_we/eval/eval_data.txt"
    datas = open(data_path).read().splitlines()
    for data in datas:
        example = json.loads(data)
        prompt = example["prompt"]
        best = example["best"]
        bad = example["bad"]
        prompt_list = [prompt + "[UNUSED1]" + bad.replace("\n", "<n>").replace("<|endofpiece|>", ""),
                       prompt + "[UNUSED1]" + best.replace("\n", "<n>")]
        prompt_list = [i[:500] for i in prompt_list]
        inputs = tokenizer(prompt_list, max_length=512, return_tensors="pt", padding=True)
        RM_batch = [torch.tensor(inputs["input_ids"]).numpy(), torch.tensor(inputs["attention_mask"]).numpy()]
        inputs.append(httpclient.InferInput('input_ids', list(RM_batch[0].shape), 'INT64'))
        inputs.append(httpclient.InferInput('attention_mask', list(RM_batch[1].shape), 'INT64'))
        inputs[0].set_data_from_numpy(RM_batch[0])
        inputs[1].set_data_from_numpy(RM_batch[1])
        output = httpclient.InferRequestedOutput('output')
        # try:
        results = triton_client.infer(
            "RM_model_onnx",
            inputs,
            model_version='1',
            outputs=[output],
            request_id='1'
        )
        results = results.as_numpy('output')
        rewards = [torch.tensor(results[i][0]) for i in range(len(results))]
        print(rewards)
        # 模型输入数据（bytes）


if __name__ == '__main__':
    request(1)
