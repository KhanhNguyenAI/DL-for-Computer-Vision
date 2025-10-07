import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from PIL import Image

import animals_train
def get_args():
    parser = argparse.ArgumentParser(description="Params of CNN training")
    parser.add_argument("-i", "--image_path", type=str, default="cat.jpg")
    args = parser.parse_args()
    return  args
## gpu => model,images,labels
def infer(args):
    name_classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    model = resnet18(weights= None) # wegihts = best.pt of weights
    model.fc = nn.Linear(in_features=512, out_features=10)
    checkpoint = torch.load("best.pt")
    model.load_state_dict(checkpoint["model"]) # load weights / loss

    model.to(device)
    #inference mode / evaluation mode
    """
    no : DropOut + BatchNorm
    """
    model.eval()
    """
    test image must :
1 :R G B
2 :Tensor [0,1]
3 :B C H W
    """
    image = cv2.imread(args.image_path)  # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
    #np
    image = np.transpose(image,(2,0,1))/255 # C,H,W [0,1]
    image = torch.from_numpy(image)
    image = image[None,:,:,:].to(device).float() # np add bathSize


    #ToTensor of torch
    #image = Image.fromarray(image)  # Convert to PIL
    #image = ToTensor()(image)  # Now it's a tensor: (C, H, W), float32 in [0,1]

    #image = image.unsqueeze(dim=0).to(device) # add bathSize
    #image = image.float() # doubleTensor to floatTensor
    print(image.shape)
    softmax = nn.Softmax() # % of 10 classes
    with torch.no_grad() :
        output = model(image)[0]
        output = softmax(output) #%
        print(output)
        index = output.argmax() # % max in 10 predicted classes
        print(index)
        print(name_classes[index])

if __name__ == "__main__":

    args = get_args()
    infer(args)