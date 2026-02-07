import sys
import platform
import matplotlib
import numpy
import pip
import scipy

print(f"{'Package':<12} | Version")
print("-" * 25)
print(f"{'Python':<12} | {sys.version.split()[0]}")
print(f"{'OS':<12} | {platform.platform()}")
print(f"{'pip':<12} | {pip.__version__}")
print(f"{'matplotlib':<12} | {matplotlib.__version__}")
print(f"{'numpy':<12} | {numpy.__version__}")
print(f"{'scipy':<12} | {scipy.__version__}")
