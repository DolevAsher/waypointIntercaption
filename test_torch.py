# import sys
# import ctypes
#
# # Explicitly test loading the failing DLL
# dll_path = r"C:\Users\dolev\Documents\computer_science\PycharmProjects\wingman\.venv\Lib\site-packages\torch\lib\c10.dll"
#
# try:
#     ctypes.CDLL(dll_path)
#     print("SUCCESS: c10.dll loaded directly via ctypes!")
# except Exception as e:
#     print(f"FAILED loading c10.dll directly: {e}")
#
# import torch
# print(f"SUCCESS: PyTorch version {torch.__version__} loaded correctly!")

import torch
print("PyTorch version:", torch.__version__)