import torch
import torch.nn as nn
from torch.autograd import Variable
from models import test_LSTM
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from preprocess import data_split,normalize_data
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = './'

df = pd.read_csv("./STTcsv", index_col = 0)

STT = df[df.symbol == 'STT'].copy()
#print(GOOG)
STT.drop(['symbol'],1,inplace=True)
STT_new = normalize_data(STT)
#print(GOOG_new)
window = 15
X_train, y_train, X_test, y_test = data_split(STT_new, window)

INPUT_SIZE = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 1

learning_rate = 0.001
num_epochs = 50

rnn = test_LSTM(input_dim=INPUT_SIZE,hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_dim=OUTPUT_SIZE)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

rnn.to(device)
rnn.train()

for epoch in range(num_epochs):
    for inputs, label in zip(X_train,y_train):
        inputs = torch.from_numpy(inputs).float().to(device)
        label = torch.from_numpy(np.array(label)).float().to(device)
        optimizer.zero_grad()

        output =rnn(inputs) # forward   
        loss=criterion(output,label) # compute loss
        loss.backward() #back propagation
        optimizer.step() #update the parameters
    print('epoch {}, loss {}'.format(epoch,loss.item()))
# above for train

result = []
with torch.no_grad():
    for inputs, label in zip(X_test,y_test):
        inputs = torch.from_numpy(inputs).float().to(device)
        label = torch.from_numpy(np.array(label)).float().to(device)
        output =rnn(inputs)    
        result.append(output)
result =np.array(result)
# above for test


plt.plot(result,color='red', label='Prediction')
plt.plot(y_test,color='blue', label='Actual')
plt.legend(loc='best')
plt.show()
#print (X_train.shape, y_train.shape,X_test.shape,y_test.shape)
