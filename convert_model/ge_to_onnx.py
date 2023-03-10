import argparse
import os
from string import Template

import torch

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# from trl import AutoModelForSeq2SeqLMWithValueHead

model_name = "REL_model_onnx"
device = torch.device('cuda:7')


model_path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/cur_model/30630"
model_path = "/search/ai/kaitongyang/RLHF_DEBUG/PPO_trl/small_glm"
query_text = '什么人不能喝三七粉'+ "[gMASK]"
response_text = '服用三七粉期间,孕妇和儿童不宜使用。 三七粉是处方药,不是药品。 过量服用会引起中毒。'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
temp_inputs = tokenizer(query_text, return_tensors="pt", padding=True)
temp_inputs = tokenizer.build_inputs_for_generation(temp_inputs, targets=response_text, max_gen_length=512, padding=False)
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
	dynamic_axes={'input_ids': {0: 'B', 1: 'C'}, 'position_ids': {0: 'B', 1: 'C', 2: 'D'}, 'attention_mask': {0: 'B', 1: 'C', 2: 'D', 3: 'E'}}  # 声明动态维度，默认输入维度固定，可以声明为可变维度（用字符串占位符表示）
)


# traced_script_module.save(f"model_store/{model_name}/1/traced-model.pt")



