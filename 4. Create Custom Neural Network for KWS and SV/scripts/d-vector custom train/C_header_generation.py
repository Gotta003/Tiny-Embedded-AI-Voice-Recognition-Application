import numpy as np
import os

def sanitize_name(name):
    """Sanitize array name for C variable naming."""
    return name.replace('.', '_').replace('-', '_')

def generate_npz_header(npz_dir="/content/d_vectors", output_file="d_vectors.h"):
    """Generate a C header with 2D arrays (samples x 256) from .npz files"""
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    
    with open(output_file, "w") as f:
        f.write("#ifndef D_VECTORS_H\n")
        f.write("#define D_VECTORS_H\n\n")
        f.write("#include <stddef.h>\n\n")
        
        for file in sorted(npz_files):
            file_path = os.path.join(npz_dir, file)
            with np.load(file_path) as data:
                if 'd_vectors' not in data:
                    continue
                    
                d_vectors = data['d_vectors']
                if d_vectors.ndim != 2 or d_vectors.shape[1] != 256:
                    continue
                    
                array_name = sanitize_name(os.path.splitext(file)[0])
                samples = d_vectors.shape[0]
                f.write(f"// {samples} samples of 256-dimensional vectors\n")
                f.write(f"static const float {array_name}[{samples}][256] = {{\n")
                for sample in d_vectors:
                    f.write("    {")
                    line = ", ".join(f"{x:.6f}f" for x in sample)
                    f.write(line)
                    f.write("},\n")
                
                f.write("};\n\n")
        
        f.write("#endif // D_VECTORS_H\n")

if __name__ == "__main__":
    generate_npz_header()
