import tensorflow as tf
import numpy as np
import re

# Function to sanitize file names
def sanitize_filename(name):
    # Replace invalid characters (e.g., '/') with underscores
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

# Function to quantize data to 4-bit
def quantize_to_4bit(data, signed=True):
    if signed:
        # Map data to the range [-8, 7] for signed 4-bit
        min_val, max_val = -8, 7
    else:
        # Map data to the range [0, 15] for unsigned 4-bit
        min_val, max_val = 0, 15

    # Normalize data to the 4-bit range
    data = np.clip(data, min_val, max_val)
    data = np.round(data).astype(np.int8)  # Use int8 for storage

    return data

# Function to pack 4-bit values into bytes
def pack_4bit_to_bytes(data):
    # Ensure the data has an even number of elements
    if len(data) % 2 != 0:
        data = np.append(data, 0)  # Pad with 0 if necessary

    # Reshape the data into pairs of 4-bit values
    data = data.reshape(-1, 2)

    # Pack each pair into a single byte
    packed_data = (data[:, 0] << 4) | (data[:, 1] & 0x0F)
    return packed_data.astype(np.uint8)

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

# Save weights (4-bit packed) and biases (int32_t) to binary files
for name, data in weights.items():
    sanitized_name = sanitize_filename(name)
    print(f"Attempting to save weights to {sanitized_name}.bin")

    # Quantize to 4-bit
    quantized_data = quantize_to_4bit(data.flatten(), signed=True)

    # Pack 4-bit values into bytes
    packed_data = pack_4bit_to_bytes(quantized_data)

    # Save to binary file
    packed_data.tofile(f"{sanitized_name}.bin")
    print(f"Saved weights to {sanitized_name}.bin")

for name, data in biases.items():
    sanitized_name = sanitize_filename(name)
    print(f"Attempting to save biases to {sanitized_name}.bin")

    # Convert biases to int32_t
    int32_data = data.flatten().astype(np.int32)

    # Save to binary file
    int32_data.tofile(f"{sanitized_name}.bin")
    print(f"Saved biases to {sanitized_name}.bin")
