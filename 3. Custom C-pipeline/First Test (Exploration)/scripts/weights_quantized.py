import tensorflow as tf
import numpy as np
import re

# Function to sanitize file names
def sanitize_filename(name):
    # Replace invalid characters (e.g., '/') with underscores
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

# Load the TensorFlow Lite model
model_path = "/content/trained.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)

# Allocate tensors
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
        print(f"Tensor: {tensor_name}, Shape: {tensor_data.shape}, Dtype: {tensor_data.dtype}")

        # Check if the tensor is a weight or bias
        if 'MatMul' in tensor_name and 'BiasAdd' not in tensor_name:  # Weights
            weights[tensor_name] = tensor_data
        elif 'BiasAdd' in tensor_name:  # Biases
            biases[tensor_name] = tensor_data
    except ValueError as e:
        print(f"Skipping tensor {tensor_name}: {e}")

# Print weights and biases
print("\nWeights:")
for name, data in weights.items():
    print(f"{name}: Shape {data.shape}, Dtype {data.dtype}")

print("\nBiases:")
for name, data in biases.items():
    print(f"{name}: Shape {data.shape}, Dtype {data.dtype}")

# Save weights and biases to binary files
for name, data in weights.items():
    sanitized_name = sanitize_filename(name)
    print(f"Attempting to save weights to {sanitized_name}.bin")
    data.tofile(f"{sanitized_name}.bin")
    print(f"Saved weights to {sanitized_name}.bin")

for name, data in biases.items():
    sanitized_name = sanitize_filename(name)
    print(f"Attempting to save biases to {sanitized_name}.bin")
    data.tofile(f"{sanitized_name}.bin")
    print(f"Saved biases to {sanitized_name}.bin")
