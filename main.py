import torch
from src.model import multiClassModel
from src.asnDataset import create_dataset
from src.trainTools import next_epoch, test_results
import numpy as np
import sys

def main(filename):

  NUM_EPOCHS = 100
  LEARNING_RATE = 0.01
  TYPE = ["PRESOCIAL", "SUBSOCIAL", "SOLITARY BUT SOCIAL", "PARASOCIAL", "EUSOCIAL"]
  
  train_loader, test_loader = create_dataset([1,2,3,4,5],7,filename)

  print("Data Loaded!\n")

  mdl = multiClassModel()
  optimizer = torch.optim.Adam(mdl.parameters(), lr=LEARNING_RATE)
  loss_fn = torch.nn.CrossEntropyLoss()

  print("Model Created!\n")

  acc_array = []
  loss_array = []
  for i in range(NUM_EPOCHS):

      mdl.train()
      next_epoch(train_loader, mdl, loss_fn, optimizer)
      mdl.eval()

      results = test_results(test_loader, mdl, loss_fn)
      if((i+1)%5==0):
        print("Epoch %i:\n  Loss: %f\n  Accuracy: %2.2f%%"%(i+1, results[0], round(results[1]*100,2)))
        print("Accuracy By Catagory:")
        for j in range(5):
          print("  %s: %2.2f%%"%(TYPE[j],results[2][j]*100))
        print("\n")
      loss_array.append(results[0])
      acc_array.append(results[1])


if __name__ == '__main__':
  filename = sys.argv[1]
  main(filename)

