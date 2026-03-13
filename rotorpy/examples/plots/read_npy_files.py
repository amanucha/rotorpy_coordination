import numpy as np

# Load the compressed archive
data = np.load('plot_data.npz')

# Access the specific file
# Note: You can omit the .npy extension when accessing the key
if 'max_execution_time' in data:
    max_time = data['max_execution_time']
    print("Max Execution Time:", max_time)
else:
    print("File 'max_execution_time.npy' not found in the archive.")
    print("Available files:", data.files)