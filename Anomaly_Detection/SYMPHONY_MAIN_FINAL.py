import os
import copy 
import time
import wandb
import random
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from SYMPHONY_UTILS_FINAL import (CustomDataset, CosineAnnealingWarmUpRestarts, EarlyStopping, 
                                plot_history, dummy_and_add_feature)
from SYMPHONY_MODEL_FINAL import Network

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Training', add_help=False)

    # Model parameters
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--drop_out', default=0.1, type=float)

    # Optimizer parameters
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_t', default=10, type=int)
    parser.add_argument('--lr_scheduler', default='CosineAnnealingWarmUpRestarts', type=str)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--min_delta', default=1e-6, type=float)

    # Training parameters
    parser.add_argument('--train_data', default='/home/ubuntu/lsy/SYMPHONY/normal_TTT_preprocessed.csv', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--num_workers', default=24, type=int)
    parser.add_argument('--text', default='default', type=str)
    parser.add_argument('--device', default='cpu', type=str)

    return parser

def main(args):
    seed = 10
    suffix = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M")

    config = {
        # Model parameters
        'batch_size': args.batch_size,
        'hidden_size': args.hidden_size,
        'depth': args.depth,
        'drop_out': args.drop_out,
        
        # Optimizer parameters
        'optimizer': args.optimizer,
        'lr': args.lr,
        'lr_t': args.lr_t,
        'lr_scheduler': args.lr_scheduler,
        'gamma': args.gamma,
        'patience': args.patience,
        'weight_decay': args.weight_decay,
        'min_delta': args.min_delta,
        
        # Training parameters
        'train_data': args.train_data,
        'epochs': args.epochs,
        'num_workers': args.num_workers,
        'text': args.text,
        'device': args.device,
        }
    
    model_save_name='./RESULTS/'+config['text']+"_"+suffix+"("+ str(config['batch_size'])+"_"+\
                                                                str(config['hidden_size'])+"___"+\
                                                                str(config['optimizer'])+"_"+\
                                                                str(config['lr'])+"_"+\
                                                                str(config['lr_t'])+"_"+\
                                                                str(config['lr_scheduler'])+"_"+\
                                                                str(config['gamma'])+"_"+\
                                                                str(config['patience'])+"_"+\
                                                                str(config['weight_decay'])+"___"+\
                                                                str(config['epochs'])+")"
                                                            
    config['model_save_name'] = model_save_name
    print('model_save_name: '+config['model_save_name'].split("/")[-1])
    
    # -------------------------------------------------------------------------------------------

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    config['device'] = device

    # -------------------------------------------------------------------------------------------

    # Data load
    Train_df = pd.read_csv(config['train_data'])
    Train_df[['sin_second', 'cos_second', 'sin_minute', 'cos_minute', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']] = Train_df['t'].apply(dummy_and_add_feature).tolist()
    Train_df.drop(['t', 'anomaly_label'], axis=1, inplace=True)
    
    target_col = ['node5_CPU_target', 'node6_CPU_target', 'node7_CPU_target', 'node8_CPU_target',
                'node5_MEM_target', 'node6_MEM_target', 'node7_MEM_target', 'node8_MEM_target', 
                'node5_NET_lo-read-KB/s', 'node6_NET_lo-read-KB/s', 'node7_NET_lo-read-KB/s', 'node8_NET_lo-read-KB/s',
                'node5_NET_lo-write-KB/s', 'node6_NET_lo-write-KB/s', 'node7_NET_lo-write-KB/s', 'node8_NET_lo-write-KB/s']
    
    explanatory_col = Train_df.columns
    explanatory_col = list(set(explanatory_col) - set(target_col))
    
    config['explanatory_col'] = explanatory_col
    config['target_col'] = target_col
    
    # -------------------------------------------------------------------------------------------
    
    wandb_name = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M%S")
    wandb.init(project='SYMPHONY', name=wandb_name, entity="sylee1996", config=config, reinit=True)
    
    train_df, valid_df = train_test_split(Train_df, test_size=0.1, random_state=10)
    

    train_df = Train_df.iloc[train_idx]
    valid_df = Train_df.iloc[valid_idx]
    
    # Train
    train_set = CustomDataset(data=train_df, config=config)
    Train_loader=DataLoader(dataset=train_set, 
                            batch_size=config['batch_size'],
                            prefetch_factor=config['batch_size']*2,
                            num_workers=config['num_workers'],
                            pin_memory=True, 
                            shuffle=True,
                            drop_last=False)

    valid_set = CustomDataset(data=valid_df, config=config)
    Valid_loader=DataLoader(dataset=valid_set, 
                            batch_size=config['batch_size'],
                            prefetch_factor=config['batch_size']*2,
                            num_workers=config['num_workers'],
                            pin_memory=True, 
                            shuffle=True,
                            drop_last=False)
# -------------------------------------------------------------------------------------------

    model = Network(config).to(config['device'])
    model = nn.DataParallel(model).to(config['device'])

    if config['lr_scheduler'] == 'CosineAnnealingLR':
        optimizer=torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
                                
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['lr_t'], eta_min=0)

    elif config['lr_scheduler'] == 'CosineAnnealingWarmUpRestarts':
        optimizer = torch.optim.AdamW(model.parameters(), lr=0)
        
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=config['lr_t'], eta_max=config['lr'], gamma=config['gamma'], T_mult=1, T_up=0)
    
    criterion = nn.MSELoss().cuda()
    scaler = torch.cuda.amp.GradScaler() 
    early_stopping_loss = EarlyStopping(patience=config['patience'], mode='min', min_delta=config['min_delta'])

    # -------------------------------------------------------------------------------------------
    epochs = config['epochs']
    history = { 'valid_loss':[]}

    best_loss = 10
    quantile_loss_treshold_list = []
    
    wandb.watch(model)

    for epoch in range(epochs):
        
        valid_loss = 0
        val_loss_treshold_list = []
                            
        model.train()
        tqdm_dataset = tqdm(enumerate(Train_loader), total=len(Train_loader))

        for batch_id, [batch_x, batch_y] in tqdm_dataset:
            batch_x = batch_x.to(config['device'])
            batch_y = batch_y.to(config['device'])
            
            with torch.cuda.amp.autocast():
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tqdm_dataset.set_postfix({
                'Epoch' : epoch+1,
                'loss' : '{:08f}'.format(loss),
                })
        
        
        scheduler.step()
        loss_treshold_list = []
        
        model.eval()
        tqdm_valid_dataset = tqdm(enumerate(Valid_loader), total=len(Valid_loader))
        
        for val_batch_id, [val_batch_x, val_batch_y] in tqdm_valid_dataset:
            val_batch_x = val_batch_x.to(config['device'])
            val_batch_y = val_batch_y.to(config['device'])

            #valid
            val_pred = model(val_batch_x)
            val_loss = criterion(val_pred, val_batch_y)

            valid_loss += val_loss.item()
            
            tqdm_valid_dataset.set_postfix({
                'Epoch' : epoch+1,
                'loss' : '{:06f}'.format(valid_loss),
                })

            val_loss_treshold_list.append(valid_loss/len(val_batch_y))
            
        quantile_loss_treshold_list.append(np.quantile(np.array(val_loss_treshold_list), 0.75))
            
        # -------------------------------------------------------------------------------------------
        valid_losses = valid_loss/len(Valid_loader)
        history['valid_loss'].append(valid_losses)
        
        print_best = 0    
        if (history['valid_loss'][-1] <= best_loss):
            best_loss = history['valid_loss'][-1]
            
            best_idx = epoch+1
            model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict()
            best_model_wts = copy.deepcopy(model_state_dict)
            
            # load and save best model weights
            model.module.load_state_dict(best_model_wts)
            torch.save(best_model_wts, config['model_save_name'] + ".pt")
            print_best = '==> best model saved %d epoch / loss : %.8f'%(best_idx, history['valid_loss'][-1])
        
        del loss_treshold_list
        print(f'Epoch [{epoch+1}]    val_loss: {valid_losses:.8f}')
        print('\n') if type(print_best)==int else print(print_best,'\n')
        
        plt.plot(val_pred.item().cpu().numpy(),'-x', label="Pred")
        plt.plot(val_batch_y.item().cpu().numpy(), '-o', label="Real")
        plt.xlabel('index')
        plt.ylabel('Value')
        
        wandb.log({
                "Plot": plt,
                "Epoch": epoch+1,
                "loss": loss,
                "valid_loss": valid_loss,
                "best_loss": best_loss,
            })
        
        if early_stopping_loss.step(torch.tensor(history['valid_loss'][-1])):
            break
    
    plot_history(history, best_idx-1, config)
    print('Valid loss 75% quantile is {}!'.format(np.mean(quantile_loss_treshold_list)))

    file = open('{}_{}.txt'.format(config['model_save_name']), 'w')
    file.write(str(np.mean(quantile_loss_treshold_list)))
    file.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    start = time.time()
    main(args)
    print("time :", time.time() - start)