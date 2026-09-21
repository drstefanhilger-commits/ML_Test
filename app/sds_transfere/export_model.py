import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load("hbd_model.pkl")

initial_type = [('input', FloatTensorType([None, 12]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("hbd_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
