import argparse
import os
from string import Template

import torch

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# from trl import AutoModelForSeq2SeqLMWithValueHead

model_name = "REL_model_onnx"
device = torch.device('cuda:7')


model_path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/cur_model/30630"
model_path = "./model/GLM-10B-chinese-customization_02-28-11-26/30630/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
inputs = tokenizer("reward model's hash"+"[gMASK]", return_tensors="pt")
ge_inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
temp_inputs = tokenizer.build_inputs_for_generation_from_tensor(inputs['input_ids'], inputs["input_ids"])
print(temp_inputs)

model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torchscript=True, trust_remote_code=True, return_dict=False)
model = model.eval()  # 转换为eval模式
# print(model(**temp_inputs))
inputs = (temp_inputs['input_ids'], temp_inputs['position_ids'], temp_inputs['attention_mask'])  # 模型测试输入数据
os.makedirs(f"model_store/{model_name}/1", exist_ok=True)
torch.onnx.export(
	model,
	inputs,
	f"model_store/{model_name}/1/model.onnx",  # 输出模型文件名
	input_names=['input_ids', 'position_ids', 'attention_mask'],  # 输入节点名，每一个名称对应一个输入名
    output_names=['output'],  # 输出节点名，每一个名称对应一个输出名
	opset_version=11,
	use_external_data_format=True,
	dynamic_axes={'input_ids': {0: 'B', 1: 'C'}, 'position_ids': {0: 'B', 1: 'C', 2: 'D'}, 'attention_mask': {0: 'B', 1: 'C', 2: 'D', 3: 'E'}}  # 声明动态维度，默认输入维度固定，可以声明为可变维度（用字符串占位符表示）
)


# traced_script_module.save(f"model_store/{model_name}/1/traced-model.pt")



