import torch
from torch.utils.data import DataLoader
from typing import Callable
import os

def train_classifier(
        net:torch.nn.Module,
        opt:torch.optim.Optimizer,
        trainloader:DataLoader,
        valloader:DataLoader,
        criterion:Callable,
        epochs:int,
        print_every=10,
        model_name="net",
        save_path="./tmp/"):
    
    train_loss = []
    val_loss = []
    for epoch in range(epochs):

        # Train
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            if i % print_every == (print_every-1):
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.3f}')
        
        train_loss.append(running_loss/(i+1))
        
        # Validate
        running_loss = 0.0
        for i, data in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            running_loss += criterion(outputs, labels).item()
        val_loss.append(running_loss/(i+1))

        # Save
        if len(val_loss)==1 or val_loss[-1] < val_loss[-2]:
            print("Validation Loss decreased, Saving")
            torch.save(net.state_dict(), os.path.join(save_path,f"{model_name}.pth"))

    return train_loss,val_loss


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp