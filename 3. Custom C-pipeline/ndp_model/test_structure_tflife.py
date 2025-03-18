import tensorflow as tf

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
    tensor_data = interpreter.get_tensor(tensor['index'])
    
    if 'kernel' in tensor_name:  # Weights
        weights[tensor_name] = tensor_data
    elif 'bias' in tensor_name:  # Biases
        biases[tensor_name] = tensor_data

# Print weights and biases
print("Weights:", weights)
print("Biases:", biases)
