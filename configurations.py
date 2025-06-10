import torch
from torchvision import transforms
import os


# file paths
PATH_TO_FOLDER = os.getcwd()
# best test_accuracy
# (needs manual adjustment at each start)
BEST_SO_FAR = 0.0
# learning parameters
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
# dataloader settings
# only needs to be true once to townload the dataset
DOWNLOAD_DATASET = True
SHUFFLE = True
NUM_WORKERS = 4
PIN_MEMORY = True
# for optimizing image this needs to be one
DROP_LAST = True
NUM_EPOCHS = 100
# training settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_FILE = "96.1 top checkpoint.pth.tar"
CHECK_FILE_NAME = "checkpoint.pth.tar"
SAVE = True
LOAD = False
# flag for training the model or optimizing an image
TO_OPTIMIZE_IMPUT = [
    0,  # 0
    0,  # 1
    0,  # 2
    0,  # 3
    0,  # 4
    0,  #  5
    0,  # 6
    1,  #  7
    0,  # 8
    0,  # 9
]
TO_OPTIMIZE_IMPUT = torch.tensor(TO_OPTIMIZE_IMPUT).unsqueeze(0).to(torch.float32)
# saving an input image
IMAGE_FILE = "input.pt"
LOAD_IMAGE = False
NUM_EPOCHS_IMAGE = 2000
LEARNING_RATE_IMAGES = 1e-4
# image loading
TRANSFORMATION = transforms.Compose([transforms.ToTensor()])
