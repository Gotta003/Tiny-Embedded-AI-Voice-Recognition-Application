import numpy as np
import tensorflow as tf

def dequantize_tensor(tensor, scale, zero_point):
    """Dequantize a tensor using its scale and zero-point."""
    return (tensor.astype(np.float32) - zero_point) * scale

model_path = "trained.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
details = interpreter.get_tensor_details()

weights = {}
biases = {}

for detail in details:
    try:
        tensor = interpreter.get_tensor(detail['index'])
        if "MatMul" in detail['name'] and "sequential" in detail['name']:
            if 'quantization_parameters' in detail:
                scale = detail['quantization_parameters']['scales'][0]
                zero_point = detail['quantization_parameters']['zero_points'][0]
                weights[detail['name']] = {
                    'data': tensor,
                    'scale': scale,
                    'zero_point': zero_point,
                    'dequantized': dequantize_tensor(tensor, scale, zero_point)
                }
            else:
                print(f"Skipping {detail['name']}: No quantization parameters found.")
        elif "BiasAdd" in detail['name'] and "sequential" in detail['name']:
            if 'quantization_parameters' in detail:
                scale = detail['quantization_parameters']['scales'][0]
                zero_point = detail['quantization_parameters']['zero_points'][0]
                biases[detail['name']] = {
                    'data': tensor,
                    'scale': scale,
                    'zero_point': zero_point,
                    'dequantized': dequantize_tensor(tensor, scale, zero_point)
                }
            else:
                print(f"Skipping {detail['name']}: No quantization parameters found.")
    except ValueError as e:
        print(f"Error accessing tensor {detail['name']}: {e}")

np.save("weights.npy", weights)
np.save("biases.npy", biases)

print("Weights and biases extraction Ended.")
