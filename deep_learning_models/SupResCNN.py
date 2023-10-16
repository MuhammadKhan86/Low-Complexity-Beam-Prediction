import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Define Hyper Parameters
learningRate    = 0.0001
batchSize       = 128
nEpochs         = 100
nInChannels     = 1
dataSplit       = [0.7, 0.1, 0.2]

class BMDataset(Dataset):
    def __init__(self, inputData, inputLabels):
        self.data   = (inputData-torch.mean(inputData))/torch.std(inputData)
        self.labels = inputLabels

    def __len__(self):
        # this should return the size of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.data[idx].reshape(nInChannels, 4, 4)
        target = self.labels[idx]
        return features, target

# set up DataLoader for data set
dataset = BMDataset(inputData, inputLabels)

lengths = [int(p * len(dataset)) for p in dataSplit]
# trainSet, validSet, testSet = random_split(dataset, lengths, torch.Generator().manual_seed(42))
trainSet = torch.utils.data.Subset(dataset, range(lengths[0]))
validSet = torch.utils.data.Subset(dataset, range(lengths[0],lengths[0]+lengths[1]))
testSet = torch.utils.data.Subset(dataset, range(lengths[0]+lengths[1],lengths[0]+lengths[1]+lengths[2]))

trainLoader     = DataLoader(dataset=trainSet, batch_size=batchSize, drop_last=True, shuffle=False)
validLoader     = DataLoader(dataset=validSet, batch_size=batchSize, drop_last=True, shuffle=False)
testLoader      = DataLoader(dataset=testSet,  batch_size=batchSize, drop_last=True)

class EchigoCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.upsample   = nn.Upsample(scale_factor=2)
        self.conv0      = nn.Conv2d(in_channels=nInChannels, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        self.conv1      = nn.Conv2d(in_channels=nInChannels, out_channels=4, kernel_size=(3, 3), stride=1, padding=1)
        self.act1       = nn.ReLU()
        self.conv2      = nn.Conv2d(in_channels=4,  out_channels=40, kernel_size=(3, 3), stride=1, padding=1)
        self.act2       = nn.ReLU()
        self.conv3      = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(3, 3), stride=1, padding=1)
        self.act3       = nn.ReLU()
        self.flat       = nn.Flatten()
        self.fc4        = nn.Linear(in_features=2560,   out_features=128)
        self.fc5        = nn.Linear(in_features=128,    out_features=64)
        self.sofMax     = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv0(x)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.flat(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.sofMax(x)
        return x


model = EchigoCNN()

# Define Loss and Optimization
criterion       = nn.MSELoss()
optimizer       = torch.optim.Adam(model.parameters(), lr=learningRate)

trainLossHistory    = []
trainAccHistory     = []
valLossHistory      = []
valAccHistory       = []
testLossHistory     = []
testAccHistory      = []

# Loop through the number of epochs
for epoch in range(nEpochs):
    trainLoss   = 0.0
    trainAcc    = 0.0
    valLoss     = 0.0
    valAcc      = 0.0
    testLoss    = 0.0
    testAcc     = 0.0
    # set model to train mode
    model.train()
    # iterate over the training data
    for inputs, labels in trainLoader:
        optimizer.zero_grad()
        estLabels = model(inputs)
        #compute the loss
        loss = criterion(estLabels, labels)
        loss.backward()
        optimizer.step()
        # increment the running loss and accuracy
        trainLoss   += loss.item()
        trainAcc    += (torch.argmax(estLabels, 1) == torch.argmax(labels, 1)).float().sum()

    # calculate the average training loss and accuracy
    trainLoss /= len(trainLoader)
    trainLossHistory.append(trainLoss)
    trainAcc /= len(trainLoader.dataset)
    trainAccHistory.append(trainAcc)

    for inputs, labels in validLoader:
        optimizer.zero_grad()
        predLabels = model(inputs)
        loss = criterion(predLabels, labels)
        loss.backward()
        optimizer.step()
        valLoss += loss.item()
        valAcc += (torch.argmax(predLabels, 1) == torch.argmax(labels, 1)).float().sum()

    # calculate the average validation loss and accuracy
    valLoss /= len(validLoader)
    valLossHistory.append(valLoss)
    valAcc /= len(validLoader.dataset)
    valAccHistory.append(valAcc)

    # set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for inputs, labels in testLoader:
            outLabels = model(inputs)
            loss = criterion(outLabels, labels)
            testLoss += loss.item()
            testAcc += (torch.argmax(outLabels, 1) == torch.argmax(labels, 1)).float().sum()

        # calculate the average Test loss and accuracy
        testLoss /= len(testLoader)
        testLossHistory.append(testLoss)
        testAcc /= len(testLoader.dataset)
        testAccHistory.append(testAcc)

trainInputIdx  = torch.tensor(trainSet.indices)
validInputIdx  = torch.tensor(validSet.indices)
testInputIdx   = torch.tensor(testSet.indices)

testLoader = DataLoader(dataset=testSet,  batch_size=len(testSet), drop_last=True)
with torch.no_grad():
    for inputs, labels in testLoader:
        predictions = model(inputs)
predictionsNp  = predictions.detach().cpu().numpy()

trainIdxNp   = trainInputIdx.detach().cpu().numpy()
validIdxNp   = validInputIdx.detach().cpu().numpy()
testIdxNp    = testInputIdx.detach().cpu().numpy()

    
trainLossHistoryNp = torch.Tensor(trainLossHistory).detach().cpu().numpy()
valLossHistoryNp = torch.Tensor(valLossHistory).detach().cpu().numpy()
testLossHistoryNp = torch.Tensor(testLossHistory).detach().cpu().numpy()

trainAccHistoryNp = torch.Tensor(trainAccHistory).detach().cpu().numpy()
valAccHistoryNp = torch.Tensor(valAccHistory).detach().cpu().numpy()
testAccHistoryNp = torch.Tensor(testAccHistory).detach().cpu().numpy()
