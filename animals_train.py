#data
import os.path
from turtledemo.penrose import start

from torchvision.transforms import ToTensor,Compose,Resize
from torch.utils.data import DataLoader
from animals_datasets import SetUp
from animals_model import SummaryCNN

#train
from torch import nn
import torch

# pretrained-model
from torchvision.models import resnet18, ResNet18_Weights

#eval
from tqdm.autonotebook import tqdm # loading effect
import numpy as np
from sklearn.metrics import accuracy_score


#for terminal input user
from torchsummary import summary
import argparse #can change params in terminal/cmd/server

#save/checkpoint
from  torch.utils.tensorboard import SummaryWriter #tensorboard --logdir my_tensorboard
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="PRGn")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_args():
    parser = argparse.ArgumentParser(description="Params of CNN training")
    parser.add_argument("-d", "--data_path", type=str, default="./animals", help="./animals")
    parser.add_argument("-l", "--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="epochs")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("-t", "--tensorboard", type=str, default="my_tensorboard", help="my tensorboard")
    parser.add_argument("-r","--resume",type = bool , default= False , help = 'resume train saved model')
    args = parser.parse_args()
    return  args
## gpu => model,images,labels
def train(args):

    transforms = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    train_dataset = SetUp(root= args.data_path, isTrain=True, transform=transforms)
    test_dataset = SetUp(root= args.data_path, isTrain=False, transform=transforms)

    train_batch = DataLoader(dataset=train_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4)
    test_batch = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                            num_workers=4)  ## test or val can't drop or shuffle every thing is default

    #model = SummaryCNN(num_classes=len(train_dataset.name_classes))
    model = resnet18(weights = ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    # input(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.resume  :
        checkpoint = model.load('model.pt')
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint['best_acc']
    else :
        start_epoch = 0
        best_acc = -1
    print("Training on:", device)
    print(f"data_path:{args.data_path} lr :{args.lr} batch:{args.batch_size} epochs:{args.epochs} momentum:{args.momentum} " )
    num_iter_per_epoch = len(train_batch)
    if os.path.isdir(args.tensorboard) :
        shutil.rmtree(args.tensorboard) #if folder had a file, delete it
    os.makedirs(args.tensorboard) #create new
    writer = SummaryWriter(args.tensorboard)

    for epoch in ( start_epoch,args.epochs ):
        #training
        model.train() # training mode
        progress_bar = tqdm(train_batch, colour="cyan")
        loss_ls = []
        for iteration, (images, labels) in enumerate(progress_bar):
            # Forward #local gradient
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            loss = criterion(predictions, labels)

            loss_ls.append(loss.item())
            avg_loss = np.mean(loss_ls)
            progress_bar.set_description("Train Epoch: {}/{}. Loss: {:.4f}".format(epoch + 1,args.epochs, avg_loss))

            #scalar : write num
            writer.add_scalar("Train/loss",avg_loss,global_step=epoch*num_iter_per_epoch+iteration)

            # Backward #upstream gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        #validation
        model.eval()
        epoch_loss_total = []
        labels_lst = []
        predictions_lst = []
        progress_bar = tqdm(test_batch, colour="green")
        with torch.no_grad():  # not calculate local gradient ( eval neednt backward to calculate upstream gradient)
            for iteration, (images, labels) in enumerate(progress_bar):
            # forward
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                predicted_labels = torch.argmax(predictions, dim=1)

            # calculate loss
                loss = criterion(predictions, labels)

            # collect metrics
                epoch_loss_total.append(loss.item())
                labels_lst.extend(labels.cpu().numpy().tolist())
                predictions_lst.extend(predicted_labels.cpu().numpy().tolist())


        loss_avg = np.mean(epoch_loss_total)
        # Evaluate after epoch

        acc = accuracy_score(labels_lst, predictions_lst)
        print(f"\nEpoch {epoch + 1} - Average Loss: {loss_avg:.4f}, Accuracy: {acc*100:.2f}%\n")

        #save check point
        writer.add_scalar("test/AVGloss", loss_avg, global_step=epoch)
        writer.add_scalar("test/accuracy",acc,global_step=epoch)
        plot_confusion_matrix(writer, confusion_matrix(labels_lst,predictions_lst),train_dataset.name_classes,epoch)
        #save epoch 1 +,,,,, epoch n
        #save best and last checkpoint
        checkpoint = {
            "model" : model.state_dict(), # loss of model
            "optimizer" : optimizer.state_dict(), #lr of model
            "epoch" : epoch ,#stopped epoch
            "best_acc" : best_acc
        }

        torch.save(checkpoint,"model.pt")
        if acc > best_acc :
            torch.save(checkpoint,"best.pt") # model.state_dict() or checkpoint
            best_acc = acc

def test():
    pass


if __name__ == "__main__":
    args = get_args()
    train(args)
