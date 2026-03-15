import sys
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python script.py <path/to/file.npz>")
    sys.exit(1)

data = np.load(sys.argv[1])

if 'max_execution_time' in data:
    max_time = data['max_execution_time']
    print("Max Execution Time:", max_time)
else:
    print("File 'max_execution_time.npy' not found in the archive.")
    print("Available files:", data.files)
