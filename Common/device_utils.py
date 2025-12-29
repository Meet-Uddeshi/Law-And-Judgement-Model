import torch

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[DEVICE] Using CPU (no GPU available)")
    return dev

def is_gpu_available() -> bool:
    return torch.cuda.is_available()
