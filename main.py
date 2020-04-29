import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

def train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=20):
    def train(model, train_loader,optimizer,criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)  #36
            optimizer.zero_grad()
            outputs = model(inputs)    #36*1
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()


    def valid(model, valid_loader,criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader,optimizer,criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, 'best_model.pt')

def weights_init(m):                                               # 1
    classname = m.__class__.__name__                               # 2
    if classname.find('Conv') != -1:                               # 3
        nn.init.normal_(m.weight.data, 0.0, 0.02)                  # 4
    elif classname.find('BatchNorm') != -1:                        # 5
        nn.init.normal_(m.weight.data, 1.0, 0.02)                  # 6
        nn.init.constant_(m.bias.data, 0)                          # 7

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    ## about model
    num_classes = 20

    ## about data
    data_dir = "../data/"
    inupt_size = 224
    batch_size = 36

    ## about training
    # num_e
    num_epochs = 100
    lr = 0.001

    ## model initialization
    model = models.model_B(num_classes=num_classes)
    model.apply(weights_init)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir,input_size=inupt_size, batch_size=batch_size)

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    ## loss function
    criterion = nn.CrossEntropyLoss()
    train_model(model,train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)

#5 GPU
# training: 0.3035, 0.9075
#validation: 0.7035, 0.8162

#1 GPU
#training: 0.1601, 0.9542
#validation: 0.7024, 0.8162

# Implementation
#training: 0.6322, 0.8022
#validation: 1.7913, 0.5588

# training: 0.1837, 0.9359
# validation: 0.7356, 0.8162
