from onnx import load_model, save_model
import torch
import torch.nn as nn
from onnxmltools.utils import float16_converter
import numpy as np
import onnxruntime as rt
from torch.nn.modules.upsampling import UpsamplingNearest2d
import sys
import os
model_name = sys.argv[1]

output_onnx_name = f"model_store/{model_name}/1/model.onnx"
os.makedirs(f"model_store/{model_name}_fp16/1", exist_ok=True)

onnx_model = load_model(output_onnx_name, load_external_data=False)
trans_model = float16_converter.convert_float_to_float16(onnx_model ,keep_io_types=True)
save_model(trans_model, f"model_store/{model_name}_fp16/1")
