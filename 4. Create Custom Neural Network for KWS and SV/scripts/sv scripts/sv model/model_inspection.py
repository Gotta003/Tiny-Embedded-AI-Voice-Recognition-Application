import tensorflow as tf
import numpy as np
from tabulate import tabulate

def inspect_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    ops = interpreter._get_ops_details()
    
    print("\n" + "="*50)
    print(f"Model Inspection: {model_path}")
    print("="*50 + "\n")

    # 1. Print Input/Output Details
    print("\n[Input Tensors]")
    input_table = []
    for i, detail in enumerate(input_details):
        input_table.append([
            i, detail['name'], detail['shape'], detail['dtype'],
            detail.get('quantization', 'N/A')
        ])
    print(tabulate(input_table, headers=["Index", "Name", "Shape", "DType", "Quantization"]))

    print("\n[Output Tensors]")
    output_table = []
    for i, detail in enumerate(output_details):
        output_table.append([
            i, detail['name'], detail['shape'], detail['dtype'],
            detail.get('quantization', 'N/A')
        ])
    print(tabulate(output_table, headers=["Index", "Name", "Shape", "DType", "Quantization"]))

    print("\n[All Tensors]")
    tensor_table = []
    for tensor in tensor_details:
        tensor_table.append([
            tensor['index'], tensor['name'], tensor['shape'], tensor['dtype'],
            tensor.get('quantization', 'N/A'), tensor.get('sparsity_parameters', 'N/A')
        ])
    print(tabulate(tensor_table, headers=["Index", "Name", "Shape", "DType", "Quantization", "Sparsity"]))

    print("\n[Operations]")
    ops_table = []
    for op in ops:
        ops_table.append([
            op['index'], op['op_name'], op['inputs'], op['outputs']
        ])
    print(tabulate(ops_table, headers=["Index", "Op Name", "Inputs", "Outputs"]))

    print("\n[Model Summary]")
    print(f"- Inputs: {len(input_details)}")
    print(f"- Outputs: {len(output_details)}")
    print(f"- Tensors: {len(tensor_details)}")
    print(f"- Operations: {len(ops)}")

if __name__ == "__main__":
    model_path = "d-vector-extractor-256.tflite"
    inspect_tflite_model(model_path)
