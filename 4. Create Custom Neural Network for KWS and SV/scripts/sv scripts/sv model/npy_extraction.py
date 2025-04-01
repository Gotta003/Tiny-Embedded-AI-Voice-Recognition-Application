import numpy as np
import tensorflow as tf
import os

model_path = 'd-vector-extractor-256.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

output_dir = "tflite_tensors"
os.makedirs(output_dir, exist_ok=True)
tensor_details = interpreter.get_tensor_details()

for tensor in tensor_details:
    try:
        if interpreter.get_tensor(tensor['index']).size == 0:
            print(f"Skipping empty tensor: {tensor['name']} (index {tensor['index']})")
            continue
        
        tensor_data = interpreter.get_tensor(tensor['index'])
        tensor_name = tensor['name'].replace('/', '_')

        filename = f"tensor_{tensor['index']:03d}_{tensor_name}_shape-{tensor['shape']}_dtype-{tensor['dtype']}.npy"
        np.save(os.path.join(output_dir, filename), tensor_data)
        print(f"Saved: {filename}")
    
    except ValueError as e:
        print(f"Failed to extract {tensor['name']} (index {tensor['index']}): {str(e)}")
        continue

print(f"\nExtraction complete. Output directory: {output_dir}/")
