from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def convert_to_onnx(model, n_features):
    initial_type = [("input", FloatTensorType([None, n_features]))]
    return convert_sklearn(model, initial_types=initial_type)
