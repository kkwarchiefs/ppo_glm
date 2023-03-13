from onnx import load_model, save_model
import torch
import torch.nn as nn
from onnxmltools.utils import float16_converter
import numpy as np
import onnxruntime as rt
from torch.nn.modules.upsampling import UpsamplingNearest2d
import sys
output_onnx_name = sys.argv[1]
onnx_model = load_model(output_onnx_name)
trans_model = float16_converter.convert_float_to_float16(onnx_model,keep_io_types=True)
save_model(trans_model, sys.argv[1] + '.fp16')
