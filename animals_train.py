#data
import os.path

from torchvision.transforms import ToTensor,Compose,Resize
from torch.utils.data import DataLoader
from animals_datasets import SetUp
from animals_model import SummaryCNN

#train
from tqdm.autonotebook import tqdm # loading effect
from torch import nn
import torch
#eval
import numpy as np
from sklearn.metrics import accuracy_score

from torchsummary import summary
import argparse #can change params in terminal/cmd/server

#save/checkpoint
from  torch.utils.tensorboard import SummaryWriter #tensorboard --logdir my_tensorboard
import shutil



def get_args():
    parser = argparse.ArgumentParser(description="Params of CNN training")
    parser.add_argument("-d", "--data_path", type=str, default="./animals", help="./animals")
    parser.add_argument("-l", "--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="epochs")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("-t", "--tensorboard", type=str, default="my_tensorboard", help="my tensorboard")
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

    model = SummaryCNN(num_classes=len(train_dataset.name_classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print("Training on:", device)
    print(f"data_path:{args.data_path} lr :{args.lr} batch:{args.batch_size} epochs:{args.epochs} momentum:{args.momentum} " )
    num_iter_per_epoch = len(train_batch)
    if os.path.isdir(args.tensorboard) :
        shutil.rmtree(args.tensorboard) #if folder had a file, delete it
    os.makedirs(args.tensorboard) #create new
    writer = SummaryWriter(args.tensorboard)
    best_acc = -1
    for epoch in range(args.epochs):
        #training
        model.train() # training mode
        progress_bar = tqdm(train_batch, colour="cyan")
        loss_ls = []
        for iteration, (images, labels) in enumerate(progress_bar):
            # Forward
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            loss = criterion(predictions, labels)

            loss_ls.append(loss.item())
            avg_loss = np.mean(loss_ls)
            progress_bar.set_description("Train Epoch: {}/{}. Loss: {:.4f}".format(epoch + 1,args.epochs, avg_loss))

            #scalar : write num
            writer.add_scalar("Train/loss",avg_loss,global_step=epoch*num_iter_per_epoch+iteration)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        #validation
        model.eval()
        epoch_loss_total = []
        labels_lst = []
        predictions_lst = []
        progress_bar = tqdm(test_batch, colour="green")
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

        #save epoch 1 +,,,,, epoch n
        #save best and last checkpoint 
        torch.save(model.state_dict(),"model.pt")
        if acc > best_acc :
            torch.save(model.state_dict(),"best.pt")
            best_acc = acc

def test():
    pass


if __name__ == "__main__":
    args = get_args()
    train(args)