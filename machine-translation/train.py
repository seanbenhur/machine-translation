
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(object):
  def __init__(self,model,device,loss_fn=None,optimizer=None,scheduler=None):

    #set parameters
    self.model = model
    self.device = device
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.scheduler = scheduler 

  def train_step(self,dataloader):
    #set model to training mode
    self.model.train()
    loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
      #send  data to device
      inp_data = batch.src.to(device)
      target = batch.trg.to(device)

      #forward prop
      output = self.model(inp_data, target[:-1, :])
      
      # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
      # doesn't take input in that form. For example if we have MNIST we want to have
      # output to be: (N, 10) and targets just (N). Here we can view it in a similar
      # way that we have output_words * batch_size that we want to send in into
      # our cost function, so we need to do some reshapin.
      # Let's also remove the start token while we're at it
      output = output.reshape(-1, output.shape[2])
      target = target[1:].reshape(-1)

      #zero out the gradients
      self.optimizer.zero_grad()

      loss_per_epoch = self.loss_fn(output,target)
  
      #back prop
      loss_per_epoch.backward()
      # Clip to avoid exploding gradient issues, makes sure grads are
      # within proper range
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
      #sgd step
      self.optimizer.step()
      # Cumulative Metrics
      loss += (loss_per_epoch.detach().item() - loss) / (batch_idx + 1)
      
      #self.scheduler.step(loss)
    
    return loss

  def eval_step(self,dataloader):
    #set model to eval mode
    self.model.eval()
    
    loss = 0.0
    with torch.no_grad():
      for batch_idx, batch in enumerate(dataloader):
        #send  data to device
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        #forward prop
        output = self.model(inp_data, target[:-1, :])
      
        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        loss_per_epoch = self.loss_fn(output,target).item()
      
        # Cumulative Metrics
        loss += (loss_per_epoch - loss) / (batch_idx + 1)
      
    return loss

  def train(self, num_epochs, patience, train_dataloader, val_dataloader):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
      save_model = True
      if save_model:
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
      #steps
      train_loss = self.train_step(dataloader=train_dataloader)

      val_loss = self.eval_step(dataloader=val_dataloader)

      #self.scheduler.step(val_loss)
      #early stopping
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = self.model
        torch.save(best_model.state_dict(), '/content/drive/MyDrive/Models/MLT/model.pt')
        _patience = patience

      else:
        _patience -= 1

      if not _patience:
        print("Stopping early!!")
        break
      # 

      # Logging
      print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.5f}, "
            f"val_loss: {val_loss:.5f}, "
            #f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
            f"_patience: {_patience}"
            )
      
    return best_model, best_val_loss