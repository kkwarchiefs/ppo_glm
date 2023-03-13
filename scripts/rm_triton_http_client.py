# 需要安装以下环境，在线编码镜像 默认已经安装。
# pip3 install numpy
# pip3 install nvidia-pyindex
# pip3 install tritonclient[all]

import time

import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "REL_model_onnx"  # 模型目录名/venus注册模型名称
address = "10.212.207.33:8000"  # 机器地址
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
    count = 0
    good_num = 0
    bad_num = 0
    for data in datas:
        example = json.loads(data)
        prompt = example["prompt"]
        best = example["best"]
        bad = example["bad"]
        prompt_list = [prompt + "[UNUSED1]" + bad.replace("\n", "<n>").replace("<|endofpiece|>", ""),
                       prompt + "[UNUSED1]" + best.replace("\n", "<n>")]
        prompt_list = [i[:500] for i in prompt_list]
        inputs = []
        rm_inputs = tokenizer(prompt_list, max_length=512, return_tensors="pt", padding=True)
        RM_batch = [torch.tensor(rm_inputs["input_ids"]).numpy(), torch.tensor(rm_inputs["attention_mask"]).numpy()]
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
        logits = [torch.tensor(results[i][0]) for i in range(len(results))]
        if logits[0] > logits[1]:
            bad_num = bad_num + 1
            if bad_num < 10:
                print("*" * 10)
                print(prompt)
                for response, score in zip(prompt_list, logits):
                    print("=" * 5)
                    print(score)
                    print(response)
        else:
            good_num = good_num + 1
        count = count + 1
        print(logits)
    print(good_num, count)
        # 模型输入数据（bytes）

def for_once(prompt, best, bad):
    triton_client = httpclient.InferenceServerClient(url=address)
    prompt_list = [prompt + "[UNUSED1]" + bad.replace("\n", "<n>").replace("<|endofpiece|>", ""),
                   prompt + "[UNUSED1]" + best.replace("\n", "<n>")]
    prompt_list = [i[:500] for i in prompt_list]
    inputs = []
    rm_inputs = tokenizer(prompt_list, max_length=512, return_tensors="pt", padding=True)
    RM_batch = [torch.tensor(rm_inputs["input_ids"]).numpy(), torch.tensor(rm_inputs["attention_mask"]).numpy()]
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
    logits = [torch.tensor(results[i][0]) for i in range(len(results))]
    print(logits)

if __name__ == '__main__':
    for_once('推荐几套女生夏季穿搭', "白色蕾丝吊带裙+白色高跟鞋<n>2. 灰色雪纺衬衫+黑色短裤+黑色短靴<n>3. T恤+牛仔裤+帆布鞋<n>4. 短袖白T+高腰阔腿裤+平底鞋(颜色可以搭配的亮眼一些)<n>5. V领无袖背心+牛仔长裤+马丁靴<n>6. 雪纺衫+半身裙+小白鞋<n>7. 开叉连衣裙+高跟鞋<n>以上穿搭都比较日常,可以根据自己的喜好和风格进行选择。",
             "夏季穿搭的选择有很多,可以根据自己的喜好和风格进行选择。以下是几套女生夏季穿搭的建议:<n><n>1. T恤+短裤/短裙:T恤是夏天最常见的单品之一,搭配短裤或短裙可以穿出休闲、舒适的感jio~建议尝试白色、黑色等基础色系的T恤更百搭;同时可以选择一些有趣的小图案来增加时尚感。 <n>2. 雪纺衫+半身裙:雪纺衫飘逸灵动,搭配半身裙可以增加女性的柔美气质,同时也适合通勤或者日常出行穿着。建议选择浅色系或者纯色的雪纺衫更加清爽。 <n>3. 短袖衬衫+牛仔裤:短袖衬衫不仅可以在夏日里保持凉爽,还可以增添干练、知性的味道。建议选择宽松的款式,避免过于贴身导致不舒服。 <n>4. 吊带背心+半裙:吊带背心性感又清凉,搭配半裙则可以塑造出优雅迷人的形象。建议选择亮色或者简洁的样式,避免太复杂的花纹和颜色。 <n>以上四款穿搭都是比较适合女生的夏季穿搭方式,可以根据个人偏好和经验进行选择。希望可以帮助到你嗷!")
