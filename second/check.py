
import sys, platform, traceback, struct

print("Python exe:", sys.executable)
print("Platform:", platform.platform())
print("Python bits:", struct.calcsize("P") * 8)

try:
    import torch
    print("Imported torch OK")
    print("torch.__file__:", getattr(torch, '__file__', None))
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
except Exception:
    traceback.print_exc()