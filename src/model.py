import torch

"""
This is our neural network class. We create instances of this class to 
train our model.
"""
class multiClassModel(torch.nn.Module):
  def __init__ (self):

    super(multiClassModel, self).__init__()

    self.linear1 = torch.nn.Linear(5,10)
    self.batch1 = torch.nn.BatchNorm1d(10)
    self.linear2 = torch.nn.Linear(10,10)
    self.batch2 = torch.nn.BatchNorm1d(10)
    self.linear3 = torch.nn.Linear(10,5)

    self.tanh = torch.nn.Tanh()

    self.model = torch.nn.Sequential(
      self.linear1,
      self.batch1,
      self.tanh,
      self.linear2,
      self.batch2,
      self.tanh,
      self.linear3
    )

  def forward(self, x):
    logits = self.model(x.float())
    return logits

