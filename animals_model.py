from torch import nn
from torch.nn.functional import conv2d, dropout
from torch.utils.data import DataLoader

from animals_datasets import SetUp
from torchvision.transforms import ToTensor,Compose,Resize

from torchsummary import summary

#conv1d for NLP convert text into vector
#con2d for CV IMG
#conv3d for CV Video

class DetailCNN(nn.Module) :
    def __init__(self):
        super().__init__(self)
        # out channel = 2^n = amount of kernels/filters
        # kernel size 3*3, filters = 16 , padding = "same" ---> W,H = W,H
        self.conv1_1 = nn.Conv2d(in_channels=3,kernel_size=3,out_channels=16,stride =1, padding="same")
        # num_features = out_channels
        # norm save the variance and mean of this layer
        self.norm1_1= nn.BatchNorm2d(num_features=16)
        #ReLU , Leaky Relu---> down vanishing gradient
        self.act= nn.ReLU() # can you same act or create new act

        #number of channels can up or not
        # in layer conv2 = out layer conv1
        self.conv1_2 = nn.Conv2d(in_channels=16, kernel_size=3, out_channels=16, stride=1, padding="same")
        self.norm1_2 = nn.BatchNorm2d(num_features=16)

        # after Conv1 you can short the size = Maxpooling to have a receptive field
        # normally 3 or 2
        self.pool1 = nn.MaxPool2d(kernel_size=3)

        #...............        #### repeat amount CNN you want


    def forward(self,x):
        # Conv 1
        x = self.conv1_1(x)
        x = self.norm1_1(x)
        x = self.act(x)

        x = self.conv1_2(x)
        x = self.norm1_2(x)
        x = self.act(x)
        # Conv 2
        return x

class SummaryCNN(nn.Module) :
    def __init__(self,num_classes = 10):
        super().__init__()
        self.conv1_1 = self.ConvBlock(in_channels=3,out_channels=16)
        self.conv1_2 = self.ConvBlock(in_channels=16, out_channels=16)

        self.pool1=  nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = self.ConvBlock(in_channels=16, out_channels=32)
        self.conv2_2 = self.ConvBlock(in_channels=32, out_channels=32)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = self.ConvBlock(in_channels=32, out_channels=64)
        self.conv3_2 = self.ConvBlock(in_channels=64, out_channels=64)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = self.ConvBlock(in_channels=64, out_channels=128)
        self.conv4_2 = self.ConvBlock(in_channels=128, out_channels=128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = self.ConvBlock(in_channels=128, out_channels=128)
        self.conv5_2 = self.ConvBlock(in_channels=128, out_channels=128)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        # self. flatten = nn.Flatten() # or use view

        #setUP fully connected layer
        # choose your input and out put of layer
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features= 6272, out_features= 1024)
            ,nn.ReLU()
        )
        # out of previous layer = in next layer
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512)
            , nn.ReLU()
        )
        # final : out = Num_class - number of classes which you want to predict
        self.finalFLC =nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes)
            , nn.ReLU()
        )

    def ConvBlock(self,in_channels,out_channels,kernel_size=3,padding ="same",stride = 1):
        return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,kernel_size=kernel_size,out_channels=out_channels,stride =stride, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size=kernel_size)
        )
    def forward(self,x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.pool5(x)
        x = x.view(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        # x = x.view(x.shape[0],-1)

        x= self.fc1(x)
        x = self.fc2(x)
        x = self.finalFLC(x)
        return x

if __name__ == "__main__":
    # transforms = Compose([
    #     ToTensor(),
    #     Resize((224, 224))
    # ])
    # dataset = SetUp(root='./animals', isTrain=True, transform=transforms)
    model = SummaryCNN()
    # Loader = DataLoader(dataset=dataset, shuffle=True, drop_last=True, batch_size=8,num_workers=4)
    #
    # for images, labels in Loader:
    #     images = images.float()
    #     output = model(images)
    #     print(output.shape)
    #     break
    #


    summary(model, (3, 224, 224))
