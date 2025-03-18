import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="trained.tflite")
interpreter.allocate_tensors()

# Get tensor details
tensor_details = interpreter.get_tensor_details()

# Print quantization parameters for each tensor
for tensor in tensor_details:
    print(f"Tensor: {tensor['name']}")
    print(f"  Shape: {tensor['shape']}")
    print(f"  Dtype: {tensor['dtype']}")
    if 'quantization' in tensor:
        scale = tensor['quantization'][0]
        zero_point = tensor['quantization'][1]
        print(f"  Scale: {scale}")
        print(f"  Zero Point: {zero_point}")
    print()
