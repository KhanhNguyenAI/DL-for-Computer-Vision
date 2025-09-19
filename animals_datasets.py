import cv2
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor,Compose,Resize
from PIL import Image
import os

# BY KHANH NGUYEN

class SetUp(Dataset) :
    def __init__(self,root,isTrain = True,transform = None):
        self.root = root
        self.transform = transform
        if isTrain :
            folder_path = os.path.join(root,'train')
        else :
            folder_path = os.path.join(root,"test")
        # path tiến vào file folder chính chứa toàn bộ folder ảnh
        name_classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
        self.images = []
        self.labels = []
        for index, name_class in enumerate(name_classes)  :
            # đây là folder ảnh
            image_folders = os.path.join(folder_path,name_class)
            # read path image in folder "train/class/....png"
            # path_i : [1.png,2.png...]
            for path_i in (os.listdir(image_folders)) :
                # join image_folder + path_i = complete the path to image
                # "root/folder(train/test)/ folder(image) / image.png
                img_path = os.path.join(image_folders,path_i)
                self.images.append(img_path)
                self.labels.append(name_class)

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index ):
        image_index = self.images[index]
        image = Image.open(image_index).convert("RGB")
        # PIL có dạng RGBA --> RGB
        # object --> numpyarray/Tensor
        # image = cv2.imread(image_index)
        # cv2 có thứ tự là BGR --> RGB
        # numpy.array --> Tensor
        
        # size có thể khác nhau --> resize
        label = self.labels[index]
        if self.transform is not None :
            image = self.transform(image)
        return image , label

# target setUp dataset :
# return amount image you have
# return image,label at [index]
# the image has to RGB , Tensor , same Size

if __name__ == "__main__" :
    transforms = Compose([
        ToTensor(),
        Resize((224,224))
    ])
    dataset = SetUp(root='./animals', isTrain=True,transform=transforms)

    Loader  = DataLoader(dataset=dataset,shuffle=True,drop_last=True,batch_size=8)
    for images, labels in Loader:
        print(images,labels)