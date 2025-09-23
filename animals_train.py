

from torch import nn
import torch
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor,Compose,Resize

from animals_datasets import SetUp
from animals_model import SummaryCNN

from torchsummary import summary





def train():
    transforms = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    train_dataset = SetUp(root='./animals', isTrain=True, transform=transforms)
    test_dataset = SetUp(root='./animals', isTrain=False, transform=transforms)

    train_batch = DataLoader(dataset=train_dataset, shuffle=True, drop_last=True, batch_size=8, num_workers=4)
    test_batch = DataLoader(dataset=test_dataset, batch_size=8,
                            num_workers=4)  ## test or val can't drop or shuffle every thing is default to know exactly performance

    model = SummaryCNN(num_classes=len(train_dataset.name_classes))
    lr = 0.001
    epochs = 10
    momentum = 0.9
    criterion = nn.CrossEntropyLoss() #CrossEntropyLoss have a LogSoftmax
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #use gpu if you have gpu in your computer to imporve the time of caculation
    print(device)
    for epoch in range(epochs):
        for images,labels in train_batch :
            # forward
            images = images.float()
            # labels = labels.long()

            #logits/predictions/output
            predictions = model(images)
            # caculate loss
            loss = criterion(predictions,labels)

            #backward
            optimizer.zero_grad() #delete storage which save old gradient
            loss.backward()   #caculate the loss
            optimizer.step() #update gradient
            print(loss)
def test():
    pass


if __name__ == "__main__":
    train()

