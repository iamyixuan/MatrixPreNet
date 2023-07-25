import torch 
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from ..utils.losses import Losses
from ..utils.logger import Logger

class Trainer:
    '''
    Basic trainer class
    '''
    def __init__(self, net, optimizer_name, loss_name, patience=100, dual_train=False) -> None:
        self.net = net
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam
        
        self.ls_fn = Losses(loss_name)()
        
        self.patience = patience
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use CUDA device
        else:
            self.device = torch.device("cpu")

        self.dual_train = dual_train

       

    def train(self, train,
              val,
              epochs,
              batch_size,
              learning_rate,
              save_freq=10,
              model_name='test'):
        '''
        args:
            train: training dataset
            val: validation dataset
        '''
        self.now = datetime.now().strftime('%Y-%m-%d')
        if not os.path.exists(f'./checkpoints/{self.now}_{model_name}/'):
            print("Creating model saving folder...")
            os.makedirs(f'./checkpoints/{self.now}_{model_name}/')

        self.logger = Logger()

        self.net.to(self.device)
        self.net.train()
        optimizer = self.optimizer(self.net.parameters(), lr=learning_rate)
        train_loader = DataLoader(train, batch_size=batch_size)
        val_loader = DataLoader(val, batch_size=10)

        for val in val_loader:
            if self.dual_train:
                x_val, y_val = val
                y_val, adj_val = y_val
            else:
                x_val, y_val = val
            break

        print("Starts training...")
        best_val = np.inf
        curr_patience = 0
        for ep in range(epochs):
            running_loss = []
            for x_train, y_train in tqdm(train_loader):
                if self.dual_train:
                    y_train, adj_train = y_train
                optimizer.zero_grad()
                out = self.net(x_train)
                if self.dual_train:
                    batch_loss = self.ls_fn(y_train, out, adj_train)
                else:
                    batch_loss = self.ls_fn(y_train, out)
                batch_loss.backward()
                running_loss.append(batch_loss.detach().cpu().numpy())
                optimizer.step()
            with torch.no_grad():
                val_out = self.net(x_val) 
                if self.dual_train:
                    val_loss = self.ls_fn(y_val, val_out, adj_val)
                else:
                    val_loss = self.ls_fn(y_val, val_out)

            if val_loss < best_val:
                best_val = val_loss
                curr_patience = 0
            curr_patience += 1
            self.logger.record('epoch', ep+1)
            self.logger.record('train_loss', np.mean(running_loss))
            self.logger.record('val_loss', val_loss.item())
            self.logger.print()
            if ep % save_freq == 0:
                torch.save(self.net.state_dict(), f'./checkpoints/{self.now}_{model_name}/model_saved_ep_{ep}')
            
            if curr_patience > self.patience:
                torch.save(self.net.state_dict(), f'./checkpoints/{self.now}_{model_name}/model_saved_best')
                break
            self.logger.save(f'./checkpoints/{self.now}_{model_name}/')
            
    def pred(self, test_data,  checkpoint=None):
        test_loader = DataLoader(test_data, batch_size=test_data.__len__())
        for x, y in test_loader:
            self.net.eval()
            if checkpoint is not None:
                self.net.load_state_dict(torch.load(checkpoint))
            with torch.no_grad():
                out = self.net(x) 
        return y.numpy(), out.numpy()

