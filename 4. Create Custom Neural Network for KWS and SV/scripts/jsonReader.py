import json

# Load the JSON data
with open('deployment-metadata.json') as f:
    data = json.load(f)

# Extract class names
classes = data['classes']

# Generate C code
c_code = f"static const char *class_names[] = {{\n"
for class_name in classes:
    c_code += f'    "{class_name}",\n'
c_code += "};\n\n"
c_code += f"static const int num_classes = {len(classes)};\n"

# Save to a header file
with open('./include/class_names.h', 'w') as f:
    f.write(c_code)

print("Generated 'class_names.h' for C.")
