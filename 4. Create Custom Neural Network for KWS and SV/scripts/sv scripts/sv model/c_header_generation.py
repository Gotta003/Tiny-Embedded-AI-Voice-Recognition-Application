import numpy as np
import os

def sanitize_name(name):
    """Convert tensor names to valid C identifiers."""
    return name.replace('/', '_').replace(';', '_').replace(':', '_')

def estimate_name(name):
  match name:
    case "serving_default_input_0":
        return "default_input"
    case "arith.constant":
        return "conv_4_BiasAdd_ReadVariableOp"
    case "arith.constant1":
        return "conv_3_BiasAdd_ReadVariableOp"
    case "arith.constant2":
        return "conv_2_BiasAdd_ReadVariableOp"
    case "arith.constant3":
        return "conv_1_BiasAdd_ReadVariableOp"
    case "arith.constant4":
        return "conv_4_Weights"
    case "arith.constant5":
        return "conv_3_Weights"
    case "arith.constant6":
        return "conv_2_Weights"
    case "arith.constant7":
        return "conv_1_Weights"
    case "arith.constant8":
        return "reshape_Dimensions"
    case "d-vector-extractor-256_1_batch_normalization_1_batchnorm_mul":
        return "batch_norm_mul"
    case "d-vector-extractor-256_1_batch_normalization_1_batchnorm_sub":
        return "batch_norm_sub"

def generate_header(npy_dir="npy", output_file="d_vector_extractor.h"):
    """Generate a C header from .npy files."""
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    
    with open(output_file, "w") as f:
        f.write("#ifndef D_VECTOR_EXTRACTOR_H\n")
        f.write("#define D_VECTOR_EXTRACTOR_H\n\n")

        for file in sorted(npy_files):
            data = np.load(os.path.join(npy_dir, file))
            if data.size == 0:
                continue

            tensor_name = "_".join(file.split('_')[2:-2])
            c_name = sanitize_name(tensor_name)
            c_name = estimate_name(c_name)
            f.write(f"static const float {c_name}[{data.size}] = {{\n")
            
            flat_data = data.flatten()
            for i in range(0, len(flat_data), 8):
                line = ", ".join(f"{x:.6f}f" for x in flat_data[i:i+8])
                f.write(f"    {line},\n")
            
            f.write("};\n\n")
            f.write(f"// Shape: {data.shape}\n\n")
        f.write("#endif // D_VECTOR_EXTRACTOR_H\n")

if __name__ == "__main__":
    generate_header()
