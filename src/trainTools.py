import torch

"""
Helper function to facilitate training an epoch.

Inputs: 
dl - the dataloader for the train data
model - the model to train
loss_fn - the loss function trained over
optimizer - the object that holds update values
"""
def next_epoch(dl, model, loss_fn, optimizer):
    
    size = len(dl.dataset)
    
    for X, y in dl:
        loss = loss_fn(model(X), y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


"""
Helper function to facilitate getting overall loss and test functions

Inputs: 
dl - the dataloader for the test data
model - the model to train
loss_fn - the loss function trained over

Outputs:
loss - the average loss determined by the loss function
accuracy - the average accuracy determined by percentage of test
           data correctly classified
catagory_accuracy - an array such that catagory_accuracy[i] = accuracy of all
                    test data such that y = i
"""
def test_results(dl, model, loss_fn):
    
    loss, accuracy = 0, 0
    catagory_accuracy = [0,0,0,0,0]
    catagory_number = [0,0,0,0,0]

    model.eval()

    for X, y in dl:
        loss += loss_fn(model.forward(X), y.long()).item()/len(dl)
        accuracy += (model.forward(X).argmax(1) == y).type(torch.float).sum().item()/len(dl.dataset)
        catagory_accuracy[y.long()]+=(model.forward(X).argmax(1) == y).type(torch.float).sum().item()
        catagory_number[y.long()]+=1


    model.train()

    for i in range(5):
        catagory_accuracy[i]/=catagory_number[i]

    return loss, accuracy, catagory_accuracy


