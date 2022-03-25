import torch

"""
This is our neural network class. We create instances of this class to 
train our model.
"""
class model(torch.nn.Module):
  def __init__ (self):

    super(model, self).__init__()
    self.flatten = torch.nn.Flatten()

    self.linear1 = torch.nn.Linear(5,5)
    self.batch1 = torch.nn.BatchNorm1d(100)
    self.linear2 = torch.nn.Linear(5,5)
    self.batch2 = torch.nn.BatchNorm1d(100)
    self.linear3 = torch.nn.Linear(5,5)
    self.batch3 = torch.nn.BatchNorm1d(100)
    self.linear4 = torch.nn.Linear(5,5)

    self.tanh = torch.nn.Tanh()

    self.model = torch.nn.Sequential(
      self.flatten,
      self.linear1,
      self.batch1,
      self.tanh,
      self.linear2,
      self.batch2,
      self.tanh,
      self.linear3,
      self.batch3,
      self.tanh,
      self.linear4
    )

    def forward(self, x):
      logits = self.model(x)
      return logits