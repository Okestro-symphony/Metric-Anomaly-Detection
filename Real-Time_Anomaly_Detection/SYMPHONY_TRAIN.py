import os
import copy 
import json
import time
import random
import datetime
import argparse
import numpy as np

from tqdm import tqdm
from natsort import natsorted

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from METRIC_PREPROCESSING import preprocessing_main
from METRIC_UTILS import (CustomDataset, BucketBatchingSampler, Custom_collate_fn, CosineAnnealingWarmUpRestarts, criterion)
from METRIC_MODEL import Network


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Training', add_help=False)
    
    # Configuration settings
    parser.add_argument('--data_folder_path', default='/home/ubuntu/lsy/KYOBO/croffle/data/', type=str)
    parser.add_argument('--model_save_folder', default='/home/ubuntu/lsy/KYOBO/croffle/jobs/anomaly_detection/metric_anomaly/artifacts/', type=str)
    parser.add_argument('--phase', default='Train', type=str)
    
    
    # Model parameters
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--window_size', default=30, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--drop_out', default=0.1, type=float)


    # Optimizer parameters
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_t', default=10, type=int)
    parser.add_argument('--lr_scheduler', default='CosineAnnealingLR', type=str)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)


    # Training parameters
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--device', default='cpu', type=str)

    return parser


def main(args):
    
    start = time.time()
    seed = 10
    
    config = {
        # Configuration settings
        'data_folder_path': args.data_folder_path,
        'model_save_folder': args.model_save_folder,  
        'phase': args.phase,  
        
        # Model parameters
        'batch_size': args.batch_size,
        'hidden_size': args.hidden_size,
        'window_size': args.window_size,
        'depth': args.depth,
        'drop_out': args.drop_out,
        
        # Optimizer parameters
        'lr': args.lr,
        'lr_t': args.lr_t,
        'lr_scheduler': args.lr_scheduler,
        'gamma': args.gamma,
        'weight_decay': args.weight_decay,
        
        # Training parameters
        'epochs': args.epochs,
        'num_workers': args.num_workers,
        'device': args.device,
        }
    
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

    users_folder = os.listdir(config['data_folder_path'])
        
    # 각 사용자에 대한 손실 기준 임계값 dictionary 생성  
    quantile_loss_treshold_dict = {user:{} for user in natsorted(users_folder)}

    for user in natsorted(users_folder):

        directory_path = os.path.join(config['data_folder_path'], user)

        # 각 사용자에 대한 설정 업데이트
        config['user_name'] = user
        config['data_path'] = directory_path + '/*.csv'
        config['scaler_path'] = directory_path + '/'
        config['model_save_name_artifacts'] = config['model_save_folder']
        config['model_save_name'] = config['model_save_name_artifacts'] + user
        
        # 데이터 전처리
        _, bin_df = preprocessing_main(config)

        # 데이터 로드
        col_list = list(bin_df.columns)
        col_list.remove('host_name')  # 'host_name' 컬럼 제거
        config['input_col'] = col_list
        config['input_size'] = config['window_size'] * len(col_list)
        config['z_size'] = config['window_size'] * config['hidden_size']

        # 인스턴스별 데이터 로더 생성 후 dictionary에 저장 
        Train_data_dict = {}
        for instance in bin_df.host_name.unique():
            
            # 각 인스턴스에 대한 학습 데이터 프레임 생성
            Train_df_instance = bin_df[bin_df.host_name == instance][config['input_col'] ].reset_index(drop=True)
            windows_train = Train_df_instance.values[np.arange(config['window_size'])[None, :] + np.arange(Train_df_instance.shape[0]-config['window_size']+1)[:, None]]
            torch_train = torch.from_numpy(windows_train).float().view(([windows_train.shape[0], config['input_size']]))
            
            # 학습 데이터 세트 생성
            train_set = CustomDataset(data=torch_train)
            sampler = BucketBatchingSampler(data_source=train_set, config=config)
            collate_fn = Custom_collate_fn(config=config)
            Train_instance_loader=DataLoader(dataset=train_set,
                                        sampler=sampler,
                                        collate_fn=collate_fn,
                                        pin_memory=False, num_workers=config['num_workers'], 
                                        )

            Train_data_dict[instance] = Train_instance_loader

        # 모델 생성
        model = Network(config).to(config['device'])
        model = nn.DataParallel(model).to(config['device'])

        # 학습률 스케줄러 설정
        if config['lr_scheduler'] == 'CosineAnnealingLR':
            optimizer1=torch.optim.AdamW(list(model.module.encoder.parameters())+list(model.module.decoder1.parameters()), 
                                        lr=config['lr'],
                                        weight_decay=config['weight_decay'],)
            optimizer2=torch.optim.AdamW(list(model.module.encoder.parameters())+list(model.module.decoder2.parameters()), 
                                        lr=config['lr'],
                                        weight_decay=config['weight_decay'],)
                                    
            scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=config['lr_t'], eta_min=0)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=config['lr_t'], eta_min=0)

        # 옵티마이저와 스케줄러 설정 부분
        elif config['lr_scheduler'] == 'CosineAnnealingWarmUpRestarts':
            optimizer1 = torch.optim.AdamW(list(model.module.encoder.parameters())+list(model.module.decoder1.parameters()), lr=0)
            optimizer2 = torch.optim.AdamW(list(model.module.encoder.parameters())+list(model.module.decoder1.parameters()), lr=0)
            
            # 옵티마이저에 대한 스케줄러를 설정
            scheduler1 = CosineAnnealingWarmUpRestarts(optimizer1, T_0=config['lr_t'], eta_max=config['lr'], gamma=config['gamma'], T_mult=1, T_up=0)
            scheduler2 = CosineAnnealingWarmUpRestarts(optimizer2, T_0=config['lr_t'], eta_max=config['lr'], gamma=config['gamma'], T_mult=1, T_up=0)
    
        # 에폭과 손실 이력을 설정
        epochs = config['epochs']
        history = { 'loss1':[],
                    'loss2':[],
                    'sum_loss':[]}
        sum_of_losses = []
        
        for epoch in range(epochs):
                
            final_loss_treshold_list = []
            
            for instance_id, Train_loader in Train_data_dict.items():
                
                model.train()
                tqdm_dataset = tqdm(enumerate(Train_loader), total=len(Train_loader))
                
                loss_treshold_list = []

                for batch_id, batch in tqdm_dataset:
                    batch_x = batch.to(config['device'])
                    
                    # 첫 번째 오토인코더를 훈련
                    w1, w2, w3 = model(batch_x)
                    loss1, loss2 = criterion(batch_x, w1, w2, w3, epoch+1)
                    loss1.backward()
                    optimizer1.step()
                    optimizer1.zero_grad()
                    
                    # 두 번째 오토인코더를 훈련
                    w1, w2, w3 = model(batch_x)
                    loss1, loss2 = criterion(batch_x, w1, w2, w3, epoch+1)
                    loss2.backward()
                    optimizer2.step()
                    optimizer2.zero_grad()
                    
                    # 손실값을 업데이트하고, 최종 손실값을 계산
                    loss1 += loss1.item()
                    loss2 += loss2.item()
                    
                    w1, w2, w3 = model(batch_x)
                
                    # 계산한 손실을 기반으로 최종 손실을 계산
                    final_loss = 0.5*torch.mean((batch_x-w1)**2,axis=1)+0.5*torch.mean((batch_x-w3)**2,axis=1)
                    loss_treshold_list.extend(final_loss)
                    val_loss_treshold_list = list(np.concatenate([torch.stack(loss_treshold_list[:-1]).flatten().detach().cpu().numpy(), loss_treshold_list[-1].flatten().detach().cpu().numpy()]))
                    final_loss_treshold_list.extend(val_loss_treshold_list)
                    
                # 각 에폭이 끝날 때마다 스케줄러의 step을 실행
                scheduler1.step()
                scheduler2.step()
                
            # 전체 손실을 계산하고 이력에 저장
            loss1 = loss1/(len(Train_loader)*len(Train_data_dict))
            loss2 = loss2/(len(Train_loader)*len(Train_data_dict))
            history['loss1'].append(loss1)
            history['loss2'].append(loss2)
            history['sum_loss'].append(loss1+loss2)

            # 최적의 모델 가중치를 저장
            model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict()
            best_model_wts = copy.deepcopy(model_state_dict)
            
            model.module.load_state_dict(best_model_wts)
            torch.save(best_model_wts, config['model_save_name'] + ".pt")
            quantile_loss_treshold_dict[user][instance_id] = np.quantile(np.array(final_loss_treshold_list), 0.75)
            
            sum_of_losses.append(history['loss1'][-1]+history['loss2'][-1])
            del loss_treshold_list
            
            print(f'Epoch [{epoch+1}], loss1: {loss1:.4f}, loss2: {loss2:.4f}, sum_loss: {sum_of_losses[-1]:.4f}')
    
    # 훈련 종료 후 손실 기록을 파일로 저장
    with open('{}Loss_Quantile.txt'.format(config['model_save_name_artifacts']), 'w') as file:
        json_dict = {}
        for k, v in quantile_loss_treshold_dict.items():
            if isinstance(v, dict):
                sub_dict = {}
                for sub_k, sub_v in v.items():
                    sub_dict[str(sub_k)] = float(sub_v)
                json_dict[str(k)] = sub_dict
            else:
                json_dict[str(k)] = float(v)
        file.write(json.dumps(json_dict))

    # 훈련 시간을 측정하고 출력
    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds=sec)
    print('\nTrain Time : ',result)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('PyTorch Training', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
