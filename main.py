import torch
from src.model import multiClassModel
from src.asnDataset import create_dataset
from src.trainTools import next_epoch, test_results
import numpy as np

def main():

  NUM_EPOCHS = 20
  LEARNING_RATE = 0.01
  
  train_loader, val_loader, test_loader = create_dataset([1,2,3,4,5],9)

  mdl = multiClassModel()
  optimizer = torch.optim.Adam(mdl.parameters(), lr=LEARNING_RATE)
  loss_fn = torch.nn.CrossEntropyLoss()

  acc_array = []
  loss_array = []
  for i in range(NUM_EPOCHS):

      mdl.train()
      next_epoch(train_loader, mdl, loss_fn, optimizer)
      mdl.eval()

      results = test_results(test_loader, mdl, loss_fn)
      print("Epoch %i:\n  Loss: %f\n  Accuracy: %2.2f%%\n"%(i+1, results[0], round(results[1]*100,2)))
      loss_array.append(results[0])
      acc_array.append(results[1])


if __name__ == '__main__':
  main()

