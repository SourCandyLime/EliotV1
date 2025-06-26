import os

# Extensions to include
extensions = {'Cortex.h', 'Cortex.cpp', 'Cortex.cuh', 'Neuron.h', 'Neuron.cu', 'Neuron.h'}

# Output file
output_file = "EliotDump.txt"

with open(output_file, "w", encoding="utf-8") as out:
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                path = os.path.join(root, file)
                out.write(f"\n\n// === {path} ===\n")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        out.write(f.read())
                except UnicodeDecodeError:
                    out.write("[!!] Failed to decode file, probably binary or weird encoding.\n")

print(f"\nEliot’s brain dumped into {output_file}. Ready for your clipboard.")
