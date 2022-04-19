import torch

def next_epoch(dl, model, loss_fn, optimizer):
    
    size = len(dl.dataset)
    
    for X, y in dl:
        loss = loss_fn(model(X), y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def test_results(dl, model, loss_fn):
    
    loss, accuracy = 0, 0

    model.eval()

    for X, y in dl:
        loss += loss_fn(model.forward(X), y.long()).item()/len(dl)
        accuracy += (model.forward(X).argmax(1) == y).type(torch.float).sum().item()/len(dl.dataset)

    model.train()

    return loss, accuracy


