import os
import json
import time
import random
import datetime
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from natsort import natsorted

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from METRIC_PREPROCESSING import preprocessing_main
from METRIC_UTILS import CustomDataset, BucketBatchingSampler, Custom_collate_fn
from METRIC_MODEL import Network

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Inference', add_help=False)

    # Configuration settings
    parser.add_argument('--data_folder_path', default='/home/ubuntu/lsy/KYOBO/croffle/data/', type=str)
    parser.add_argument('--model_save_folder', default='/home/ubuntu/lsy/KYOBO/croffle/jobs/anomaly_detection/metric_anomaly/artifacts/', type=str)
    parser.add_argument('--phase', default='Test', type=str)
    
    # Model parameters
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--window_size', default=30, type=int)
    parser.add_argument('--depth', default=3, type=int)

    # Test parameters
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--device', default='cpu', type=str)


    return parser


def main(args):
    
    start = time.time()
    seed=10
        
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

        # Training parameters
        'num_workers': args.num_workers,
        'device': args.device,
        
        }

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
    
    # Read the list of user folders in the data folder
    users_folder = os.listdir(config['data_folder_path'])

    for user in natsorted(users_folder):
        
        # Construct the path to the user's directory
        directory_path = os.path.join(config['data_folder_path'], user)

        # Update the configuration parameters specific to this user
        config['user_name'] = user
        config['data_path'] = directory_path + '/*.csv'
        config['scaler_path'] = directory_path + '/'
        config['model_save_name_artifacts'] = config['model_save_folder']
        config['model_save_name'] = config['model_save_name_artifacts'] + user
        
        # Load the quantile loss threshold for this user
        threshold_path = config['model_save_name_artifacts'] + 'Loss_Quantile.txt'
        with open(threshold_path, 'r') as file:
            quantile_loss_treshold_dict = json.load(file)
        
        # Perform preprocessing on the input data
        OUTPUT_DF, bin_df = preprocessing_main(config)
        OUTPUT_DF['anomaly_score'] = 0
        
        # Get the column list for input data
        col_list = list(bin_df.columns)
        col_list.remove('host_name')
        config['input_col'] = col_list
        config['input_size'] = config['window_size'] * len(col_list)
        config['z_size'] = config['window_size'] * config['hidden_size']
        
        # Split the data into separate instances based on host_name
        Test_data_dict = {}
        Test_torch_dict = {}
        
        for instance in bin_df.host_name.unique():
            
            Test_data_instance = bin_df[bin_df.host_name == instance][config['input_col'] ].reset_index(drop=True)
            windows_test = Test_data_instance.values[np.arange(config['window_size'])[None, :] + np.arange(Test_data_instance.shape[0]-config['window_size']+1)[:, None]]
            torch_test = torch.from_numpy(windows_test).float().view(([windows_test.shape[0], config['input_size']]))
            
            test_set = CustomDataset(data=torch_test)
            sampler = BucketBatchingSampler(data_source=test_set, config=config)
            collate_fn = Custom_collate_fn(config=config)
            Test_instance_loader=DataLoader(dataset=test_set,
                                        sampler=sampler,
                                        collate_fn=collate_fn,
                                        pin_memory=False, num_workers=config['num_workers'], 
                                        )

            # Store the test data and torch tensors in dictionaries
            Test_data_dict[instance] = Test_instance_loader
            Test_torch_dict[instance] = Test_data_instance

        # Load the trained model
        model = Network(config).to(config['device'])
        model = nn.DataParallel(model).to(config['device'])
        model_dict = torch.load(config['model_save_name']+'.pt')
        model.module.load_state_dict(model_dict)

        # Perform inference for each instance
        test_w1_list, test_w3_list = [], []
        test_results = []

        alpha = 0.5
        beta = 0.5

        model.eval()
        for instance_id, Test_loader in Test_data_dict.items():
                
            OUTPUT_DF_index = OUTPUT_DF[OUTPUT_DF.host_name == instance_id].index
            tqdm_test_dataset = tqdm(enumerate(Test_loader), total=len(Test_loader))

            for batch_id, batch in tqdm_test_dataset:

                batch_x = batch.to(config['device'])
                        
                with torch.no_grad():
                    test_w1, test_w2, test_w3 = model(batch_x)
                    
                    test_w1_view = test_w1.view([test_w1.shape[0], config['window_size'], -1])
                    test_w3_view = test_w3.view([test_w2.shape[0], config['window_size'], -1])
                    
                    # 인스턴스에 대한 loss를 list로 저장
                    test_w1_list.extend(test_w1_view.detach().cpu().numpy())
                    test_w3_list.extend(test_w3_view.detach().cpu().numpy())
                    test_results.extend(0.5*torch.mean((batch_x-test_w1)**2,axis=1)+0.5*torch.mean((batch_x-test_w3)**2,axis=1))

            loss1_dic = {i:[] for i in range(len(Test_torch_dict[instance_id]))}
            loss3_dic = {i:[] for i in range(len(Test_torch_dict[instance_id]))}

            for dim_0 in range(len(Test_torch_dict[instance_id])-config['window_size']+1):
                
                for dict_idx in range(dim_0, dim_0+config['window_size']):
                    
                    loss1_dic[dict_idx].append(test_w1_list[dim_0][dict_idx - dim_0])
                    loss3_dic[dict_idx].append(test_w3_list[dim_0][dict_idx - dim_0])
                    
            new_loss_list1, new_loss_list3 = [], []

            for value1 in loss1_dic.values(): 
                new_loss_list1.append(np.mean(value1, axis=0))
            for value3 in loss3_dic.values(): 
                new_loss_list3.append(np.mean(value3, axis=0))
                
            new_df1 = pd.DataFrame(new_loss_list1, columns=col_list)
            new_df3 = pd.DataFrame(new_loss_list3, columns=col_list)

            test_loss = alpha*np.mean((Test_torch_dict[instance_id].values - new_df1.values)**2,axis=1) + beta*np.mean((Test_torch_dict[instance_id].values - new_df3.values)**2,axis=1)
            OUTPUT_DF['anomaly_score'].loc[OUTPUT_DF_index] = test_loss
            
        # Iterate over the DataFrame rows and check if the anomaly score is greater than the instance dictionary value
        anomaly_label = []
        for index, row in OUTPUT_DF.iterrows():
            host_name = row['host_name']
            anomaly_score = row['anomaly_score']
            
            if host_name in quantile_loss_treshold_dict[user] and anomaly_score > quantile_loss_treshold_dict[user][host_name]:
                anomaly_label.append(1)
            else:
                anomaly_label.append(0)
                
        OUTPUT_DF['anomaly_label'] = anomaly_label
        OUTPUT_DF.to_csv("{}_METRIC_OUTPUT.csv".format(config['scaler_path']+config['user_name']), index=False)
    
    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds=sec)
    print('\nInference Time : ',result)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Inference script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)