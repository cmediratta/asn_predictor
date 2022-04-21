import torch
from src.model import multiClassModel
from src.asnDataset import create_dataset
from src.trainTools import next_epoch, test_results
from src.perturbDataset import perturb_data
from src.analysis import graph_PCA, confusion_matrix
import numpy as np
import sys

def main(filename, train_size=0):

  NUM_EPOCHS = 100
  LEARNING_RATE = 0.01
  TYPE = ["PRESOCIAL", "SUBSOCIAL", "SOLITARY BUT SOCIAL", "PARASOCIAL", "EUSOCIAL"]
  
  if(filename[(len(filename)-13):]!="Perturbed.csv"):
    train_loader, test_loader = create_dataset(filename, train_size)
  else:
    train_loader, test_loader = create_dataset(filename, train_size, randomize=False)

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

  graph_PCA(train_loader,mdl)
  confusion_matrix(test_loader,mdl)


if __name__ == '__main__':
  filename = sys.argv[1]
  if(len(sys.argv)>2 and sys.argv[2]=="perturb"):
    t_size = perturb_data(filename)
    filename = filename[:(len(filename)-4)]+"Perturbed.csv"
    main(filename, train_size=t_size)
  else:
    main(filename)

