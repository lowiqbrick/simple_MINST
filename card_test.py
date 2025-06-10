"""
tells the user how many (if any) cuda devices are available
"""

import torch

# https://pytorch.org/get-started/locally/
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    amount_cuda = torch.cuda.device_count()
    if is_cuda:
        print(f"cuda is available; {amount_cuda} devices found")
    else:
        print("cuda is not available")
