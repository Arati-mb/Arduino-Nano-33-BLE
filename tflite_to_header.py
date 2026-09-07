# tflite_to_header.py

INPUT_FILE = "tomato_model_int8.tflite"
OUTPUT_FILE = "model.h"

print("Reading:", INPUT_FILE)

with open(INPUT_FILE, "rb") as f:
    data = f.read()

print("Model size:", len(data), "bytes")

with open(OUTPUT_FILE, "w") as f:

    f.write("#ifndef MODEL_H\n")
    f.write("#define MODEL_H\n\n")

    f.write("#include <stdint.h>\n\n")

    f.write("const unsigned char model[] = {\n")

    for i in range(0, len(data), 12):

        chunk = data[i:i+12]

        line = "  "

        for byte in chunk:
            line += "0x{:02x}, ".format(byte)

        f.write(line + "\n")

    f.write("};\n\n")

    f.write("const unsigned int model_len = ")
    f.write(str(len(data)))
    f.write(";\n\n")

    f.write("#endif // MODEL_H\n")


print("\n========================================")
print("MODEL.H CREATED SUCCESSFULLY")
print("========================================")
print("File:", OUTPUT_FILE)
print("Size:", len(data), "bytes")
print("========================================")