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

model_name = "RM_model_onnx"
device = torch.device('cuda:7')


RM_model_path = "/search/ai/kaitongyang/RLHF_DEBUG/RM/summarization_reward_model/checkpoint-58/"
RM_model_path = '/search/ai/pretrain_models/roberta-base-finetuned-jd-binary-chinese'
RM_tokenizer = AutoTokenizer.from_pretrained(RM_model_path)
RM_model = AutoModelForSequenceClassification.from_pretrained(RM_model_path, torchscript=True)
RM_model.to(device)
query_text = '什么人不能喝三七粉'
response_text = '服用三七粉期间,孕妇和儿童不宜使用。 三七粉是处方药,不是药品。 过量服用会引起中毒。'
input = RM_tokenizer(query_text, response_text, truncation=True, max_length=512, padding="max_length", return_tensors="pt").to(device)
print(input)
print(RM_model(input['input_ids'], input['attention_mask'], input['token_type_ids']))
RM_model = RM_model.eval()  # 转换为eval模式
inputs = (input['input_ids'], input['attention_mask'], input['token_type_ids'])  # 模型测试输入数据
os.makedirs(f"model_store/{model_name}/1", exist_ok=True)
torch.onnx.export(
	RM_model,
	inputs,
	f"model_store/{model_name}/1/model.onnx",  # 输出模型文件名
	input_names=['input_ids', 'attention_mask', 'token_type_ids'],  # 输入节点名，每一个名称对应一个输入名
    output_names=['output'],  # 输出节点名，每一个名称对应一个输出名
	opset_version=11,
	dynamic_axes={'input_ids': {0: 'B', 1: 'C'}, 'attention_mask': {0: 'B', 1: 'C'}, 'token_type_ids': {0: 'B', 1: 'C'}}  # 声明动态维度，默认输入维度固定，可以声明为可变维度（用字符串占位符表示）
)


# traced_script_module.save(f"model_store/{model_name}/1/traced-model.pt")



