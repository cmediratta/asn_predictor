from torch.utils.data import Dataset, DataLoader, random_split
import csv
import numpy as np
import random
import torch

"""
This class takes in our np matrix data in x and y creates a Dataset
object so we can utilize the pytorch suite of functions for data
management.

Inputs:
x - x_train data
y - y_train data
"""
class asnDataset(Dataset):

  def __init__(self, x, y):

    super(asnDataset,self).__init__()

    if(x.shape[0]!=y.shape[0]):
      raise RuntimeError('Data shape not matching.')

    self.x=x
    self.y=y

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, i):
    return self.x[i],self.y[i]


"""
This creates our actual dataset for easy interface with the main 
function.

Inputs:
params - an array of length 5 containing the indices of each feature we wish for this time.
         Index calculations assume title and y value are there.
n - the total number of possible features
"""
def create_dataset(filename, train_size, randomize=True):

  train, test = get_train_test(filename, train_size, randomize)

  train_loader = DataLoader(train, batch_size=4, shuffle=True)
  test_loader = DataLoader(test, batch_size=1, shuffle=False)

  return train_loader, test_loader



def get_train_test(filename, train_size, randomize):

  params = [1,2,3,4,5]
  n = 7

  x_all = []
  y_all = []

  with open("data/" + filename) as file:

    csvreader = csv.reader(file)

    for row in csvreader:
      param_index=0
      x = []

      for i in range(len(row)):

        if(param_index<5 and i==params[param_index]):
          x.append(float(row[i]))
          param_index+=1

      x_all.append(x)
      y_all.append(float(row[n-1]))

  data = asnDataset(np.array(x_all), np.array(y_all))

  train_num = max(int(0.8 * len(data)),train_size)
  test_num = len(data) - train_num
  if(randomize):
    train, test = random_split(data, (train_num, test_num))
  else:
    train = torch.utils.data.Subset(data, range(train_num))
    test = torch.utils.data.Subset(data, range(train_num, train_num+test_num))

  return train, test




