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

class LocationDataset(Dataset):
    def __init__(self, inputData, inputLabels):
        self.data   = (inputData-torch.mean(inputData))/torch.std(inputData)
        self.labels = inputLabels

    def __len__(self):
        # this should return the size of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.data[idx]
        target = self.labels[idx]
        return features, target

# set up DataLoader for data set
dataset = LocationDataset(inputData, inputLabels)

lengths = [int(p * len(dataset)) for p in dataSplit]
# trainSet, validSet, testSet = random_split(dataset, lengths, torch.Generator().manual_seed(42))
trainSet = torch.utils.data.Subset(dataset, range(lengths[0]))
validSet = torch.utils.data.Subset(dataset, range(lengths[0],lengths[0]+lengths[1]))
testSet = torch.utils.data.Subset(dataset, range(lengths[0]+lengths[1],lengths[0]+lengths[1]+lengths[2]))

trainLoader     = DataLoader(dataset=trainSet, batch_size=batchSize, drop_last=True, shuffle=False)
validLoader     = DataLoader(dataset=validSet, batch_size=batchSize, drop_last=True, shuffle=False)
testLoader      = DataLoader(dataset=testSet,  batch_size=batchSize, drop_last=True)

class LocationNN(torch.nn.Module):

    def __init__(self):
        super(LocationNN, self).__init__()

        self.lin1 = torch.nn.Linear(torch.Tensor.size(inputData, dim = 1), 64)
        self.act1 = torch.nn.LeakyReLU()

        self.lin2 = torch.nn.Linear(64, 64)
        self.act2 = torch.nn.LeakyReLU()

        self.lin3 = torch.nn.Linear(64, 64)
        self.act3 = torch.nn.LeakyReLU()

        self.lin4 = torch.nn.Linear(64, 64)
        self.act4 = torch.nn.LeakyReLU()

        self.lin5 = torch.nn.Linear(64, 64)
        self.prob = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        x = self.act3(self.lin3(x))
        x = self.act4(self.lin4(x))
        x = self.prob(self.lin5(x))
        return x


model = LocationNN()

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

testInput, testLabels = testLoader.dataset[:]
predictions    = model(testInput)
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
