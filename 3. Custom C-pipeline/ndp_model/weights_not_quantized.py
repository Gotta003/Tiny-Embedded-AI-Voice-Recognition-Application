import tensorflow as tf
import numpy as np
import re

# Function to sanitize file names
def sanitize_filename(name):
    # Replace invalid characters (e.g., '/') with underscores
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="trained.tflite")
interpreter.allocate_tensors()

# Get details of all tensors
tensor_details = interpreter.get_tensor_details()

# Extract weights and biases
weights = {}
biases = {}
for tensor in tensor_details:
    tensor_name = tensor['name']
    try:
        # Attempt to access the tensor data
        tensor_data = interpreter.get_tensor(tensor['index'])
        
        # Check if the tensor is a weight or bias
        if 'MatMul' in tensor_name:  # Weights
            if 'quantization' in tensor:
                scale = tensor['quantization'][0]
                zero_point = tensor['quantization'][1]
                dequantized_data = (tensor_data - zero_point) * scale
            else:
                dequantized_data = tensor_data
            weights[tensor_name] = dequantized_data
        elif 'BiasAdd/ReadVariableOp' in tensor_name:  # Biases
            if 'quantization' in tensor:
                scale = tensor['quantization'][0]
                zero_point = tensor['quantization'][1]
                dequantized_data = (tensor_data - zero_point) * scale
            else:
                dequantized_data = tensor_data
            biases[tensor_name] = dequantized_data
    except ValueError as e:
        print(f"Skipping tensor {tensor_name}: {e}")

# Print weights and biases
print("Weights:")
for name, data in weights.items():
    print(f"{name}: Shape {data.shape}")

print("\nBiases:")
for name, data in biases.items():
    print(f"{name}: Shape {data.shape}")

# Save weights and biases to binary files
for name, data in weights.items():
    sanitized_name = sanitize_filename(name)
    data.tofile(f"./weights/{sanitized_name}.bin")

for name, data in biases.items():
    sanitized_name = sanitize_filename(name)
    data.tofile(f"./weights/{sanitized_name}.bin")
