import numpy as np

# Load weights and biases
weights = np.load("weights.npy", allow_pickle=True).item()
biases = np.load("biases.npy", allow_pickle=True).item()

# Function to sanitize variable names
def sanitize_name(name):
    return name.replace('/', '_').replace(';', '_')

# Function to extract the last part of the name
def clear_name(name):
    parts = name.rsplit(";")
    return parts[-1]

# Generate the C header file
with open("model_weights.h", "w") as f:
    f.write("#ifndef MODEL_WEIGHTS_H\n")
    f.write("#define MODEL_WEIGHTS_H\n\n")

    # Write weights
    for key, value in weights.items():
        sanitized_key = sanitize_name(clear_name(key))
        dequantized_data = value['dequantized']
        if dequantized_data.size > 0:
            # Calculate the size of the array
            array_size = dequantized_data.size
            # Write the array declaration with size
            f.write(f"static const float {sanitized_key}[{array_size}] = {{\n")
            if dequantized_data.ndim == 1:
                f.write("    " + ", ".join(map(str, dequantized_data)) + ",\n")
            else:
                for row in dequantized_data:
                    f.write("    " + ", ".join(map(str, row)) + ",\n")
            f.write("};\n\n")

    # Write biases
    for key, value in biases.items():
        sanitized_key = sanitize_name(clear_name(key))
        dequantized_data = value['dequantized']
        if dequantized_data.size > 0:
            # Calculate the size of the array
            array_size = dequantized_data.size
            # Write the array declaration with size
            f.write(f"static const float {sanitized_key}[{array_size}] = {{\n")
            f.write("    " + ", ".join(map(str, dequantized_data)) + ",\n")
            f.write("};\n\n")

    f.write("#endif // MODEL_WEIGHTS_H\n")
