import os
import glob
import random
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
from time import mktime
from natsort import natsorted
from sklearn.decomposition import PCA

from SYMPHONY_UTILS_FINAL import normal_streaming_dict, abnormal_streaming_dict, dummy_and_add_feature

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Data Preprocessing', add_help=False)

    parser.add_argument('--dataset_path', default='/data/SYMPHONY/exathlon/data/**/*.csv', type=str)
    parser.add_argument('--text', default='FINAL', type=str)
    parser.add_argument('--pca', action='store_true')

    return parser


def scaling(phase, config, bin_df, scale_col):
    
    if phase == 'normal':
        # stn마다 변수별 최대, 최소값 구하기
        max_arr = bin_df[scale_col].max().values
        min_arr = bin_df[scale_col].min().values
            
        # 사전에 변수별 최대, 최소값 저장
        min_max_dict = {scale_col[i]:[min_arr[i], max_arr[i]] for i in range(len(scale_col))}

        # min-max scaling
        for col, (col_min, col_max) in min_max_dict.items():
            bin_df[col] = bin_df[col] - col_min
            bin_df[col] = bin_df[col] / (col_max-col_min)
            
        with open(config['scaler_path'], 'wb') as fw:
            pickle.dump(min_max_dict, fw)
        print("scaler is saved at {}".format(config['scaler_path']))
    
    else:
        print("scaler is loaded at {}".format(config['scaler_path']))
        with open(config['scaler_path'], 'rb') as fr:
            min_max_dict = pickle.load(fr)
    
        # min-max scaling
        for col, (col_min, col_max) in min_max_dict.items():
            bin_df[col] = bin_df[col] - col_min
            bin_df[col] = bin_df[col] / (col_max-col_min)
    
    return bin_df


def Preprocessing(csv_dict, config, col_list=None):
    gt_df = pd.read_csv(csv_dict['gt'][0])

    data_dict = {'normal': None,
                'abnormal': None}

    for phase in ['normal', 'abnormal']:
        bin_df = pd.DataFrame()
        
        for path in tqdm(csv_dict[phase], 
            total = len(csv_dict[phase]), ncols = 70, ascii = ' =', ## 바 모양, 첫 번째 문자는 공백이어야 작동
            leave = True, ## True 반복문 완료시 진행률 출력 남김. False 남기지 않음.
            ):
            
            df = pd.read_csv(path)
            
            if col_list != None:
                df = df[col_list]
                
            streaming_num = path.split('/')[-1].split("_")[0]
            df['streaming_num'] = streaming_num
                
            df.drop_duplicates(inplace=True)
            
            df['t'] = df['t'].apply((lambda x: datetime.datetime.fromtimestamp(x)))
            df['t'] = df['t'].astype(str)
            df['t'] = pd.to_datetime(df['t'])
            
            df.set_index(keys=['t'], inplace=True)

            idx = pd.date_range(df.index[0], df.index[-1], freq='s')
            df = df.reindex(idx, fill_value=0)

            df = df.reset_index()
            df['t'] = df['index'].apply((lambda x: mktime(x.timetuple()) + x.microsecond/1000000.0))
            df.drop(['index'], axis=1, inplace=True)
            
            df['anomaly_label'] = 0
            
            for idx, trace in enumerate(gt_df.trace_name):
                
                if trace in path:
                    anomaly_idx = []
                    for i in range(list(df[df['t'] == gt_df.root_cause_start[idx]].index)[0], 
                                    list(df[df['t'] == gt_df.root_cause_end[idx]].index)[0]+1):
                        anomaly_idx.append(i)
                                
                    df['anomaly_label'].loc[anomaly_idx] = 1
            
            if config['pca'] == False:
                for col in df.columns:
                    
                    if (df[col].isna().sum())/len(df) > 0.5:
                        if phase == 'normal':
                            for i in range(len(df['streaming_num'].unique())):
                                df = df.fillna(value=normal_streaming_dict[streaming_num])
                        else:
                            for i in range(len(df['streaming_num'].unique())):
                                df = df.fillna(value=abnormal_streaming_dict[streaming_num])
                    else:
                        df = df.interpolate(method='nearest', limit_direction='both')
                        
                        if phase == 'normal':
                            for i in range(len(df['streaming_num'].unique())):
                                df = df.fillna(value=normal_streaming_dict[streaming_num])
                        else:
                            for i in range(len(df['streaming_num'].unique())):
                                df = df.fillna(value=abnormal_streaming_dict[streaming_num])
                                
            bin_df = pd.concat([bin_df, df], axis=0)
            
        bin_df.drop(['streaming_num'], axis=1, inplace=True)

        bin_df = bin_df.fillna(0)    
        data_dict[phase] = bin_df
        
        del_col_list = []
        trash_col = []
        
        for col in data_dict[phase].columns:
            if len(data_dict[phase][col].unique()) < 10:
                trash_col.append(col)
            
        if config['pca'] == True:
            for col in data_dict[phase].columns:
                if 'driver_benchmark_userclicks' in col:
                    del_col_list.append(col)
                        
        sustain_col = ['node5_MEM_memtotal', 'node6_MEM_memtotal', 'node7_MEM_memtotal', 'node8_MEM_memtotal', 't', 'anomaly_label']
        trash_col_final = list(set(trash_col + del_col_list) - set(sustain_col))

        data_dict[phase].drop(trash_col_final, axis=1, inplace=True)
        

    common_col = set(list(data_dict['normal'].columns)).intersection(list(data_dict['abnormal'].columns))
    scale_col = list(set(common_col) - set(['t', 'anomaly_label']))

    data_dict['normal'] = scaling('normal', config, data_dict['normal'][common_col], scale_col).reset_index(drop=True)
    data_dict['abnormal'] = scaling('abnormal', config, data_dict['abnormal'][common_col], scale_col).reset_index(drop=True)
    
    return data_dict


def main(args):

    seed = 10

    config = {
        'dataset_path': args.dataset_path,
        'text': args.text,
        'pca': args.pca,
        }
    
    config['scaler_path'] = config['text'] + '_PCA_' + str(config['pca']) + '_SCALER_.pickle'
    
    # -------------------------------------------------------------------------------------------

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -------------------------------------------------------------------------------------------
    
    csv_list = []
    
    glob.glob(config['dataset_path'], recursive=True)
    csv_list = natsorted(glob.glob(config['dataset_path'], recursive=True))

    csv_dict = {
        'normal':[],
        'abnormal':[],
        'gt':[]
    }

    # -------------------------------------------------------------------------------------------

    for path in csv_list:
        file_name = path.split('/')[-1]
        
        if 'ground' in path:
            csv_dict['gt'].append(path)
        elif '0' in file_name.split("_")[1]:
            csv_dict['normal'].append(path)
        else:
            csv_dict['abnormal'].append(path)
                
    # -------------------------------------------------------------------------------------------
    
    if config['pca'] == True:
        DF_dict = Preprocessing(csv_dict, config)
        
        pca = PCA(n_components=19) # 주성분을 몇개로 할지 결정
        concat_df = pd.concat([DF_dict['normal'].drop(['t', 'anomaly_label'], axis=1), 
                            DF_dict['abnormal'].drop(['t', 'anomaly_label'], axis=1)], 
                            axis=0)
        printcipalComponents = pca.fit_transform(concat_df)
        principalDf = pd.DataFrame(data=printcipalComponents)
        
        normal_PCA_df = principalDf[:len(DF_dict['normal'])].reset_index(drop=True)
        normal_PCA_df = pd.concat([normal_PCA_df, DF_dict['normal'][['t', 'anomaly_label']]], axis = 1)
        normal_PCA_df['t'] = normal_PCA_df['t'].apply(lambda x: datetime.datetime.fromtimestamp(x)).astype(str)
        normal_PCA_df[['sin_second', 'cos_second', 
                        'sin_minute', 'cos_minute', 
                        'sin_hour', 'cos_hour', 
                        'sin_day', 'cos_day', 
                        'sin_month', 'cos_month']] = normal_PCA_df['t'].apply(dummy_and_add_feature).tolist()
        
        abnormal_PCA_df = principalDf[len(DF_dict['normal']):].reset_index(drop=True)
        abnormal_PCA_df = pd.concat([abnormal_PCA_df, DF_dict['abnormal'][['t', 'anomaly_label']]], axis = 1)
        abnormal_PCA_df['t'] = abnormal_PCA_df['t'].apply(lambda x: datetime.datetime.fromtimestamp(x)).astype(str)
        abnormal_PCA_df[['sin_second', 'cos_second', 
                        'sin_minute', 'cos_minute', 
                        'sin_hour', 'cos_hour', 
                        'sin_day', 'cos_day', 
                        'sin_month', 'cos_month']] = abnormal_PCA_df['t'].apply(dummy_and_add_feature).tolist()

        normal_PCA_df.to_csv("{}_PCA_{}_TRAIN.csv".format(config['text'], str(config['pca'])), index=False)
        abnormal_PCA_df.to_csv("{}_PCA_{}_TEST.csv".format(config['text'], str(config['pca'])), index=False)
        print("{}_PCA_{}_TRAIN.csv is saved!".format(config['text'], str(config['pca'])))
        print("{}_PCA_{}_TEST.csv is saved!".format(config['text'], str(config['pca'])))
        
    else: 
        col_list = []

        col_df = pd.read_csv(csv_dict['normal'][0])
        for col in col_df.columns:

            if  (('MEM' in col ) and ('node' in col)) or \
                (('CPU_ALL' in col ) and ('node' in col)) or \
                (('NET' in col ) and ('node' in col)) and ('NETPACKET' not in col):
                col_list.append(col)
                
        col_list.insert(0, 't')
        data_dict = Preprocessing(csv_dict, config, col_list=col_list)
        
        data_dict['normal']['t'] = data_dict['normal']['t'].apply(lambda x: datetime.datetime.fromtimestamp(x)).astype(str)
        data_dict['normal'][['sin_second', 'cos_second', 
                        'sin_minute', 'cos_minute', 
                        'sin_hour', 'cos_hour', 
                        'sin_day', 'cos_day', 
                        'sin_month', 'cos_month']] = data_dict['normal']['t'].apply(dummy_and_add_feature).tolist()
        
        data_dict['abnormal']['t'] = data_dict['abnormal']['t'].apply(lambda x: datetime.datetime.fromtimestamp(x)).astype(str)
        data_dict['abnormal'][['sin_second', 'cos_second', 
                        'sin_minute', 'cos_minute', 
                        'sin_hour', 'cos_hour', 
                        'sin_day', 'cos_day', 
                        'sin_month', 'cos_month']] = data_dict['abnormal']['t'].apply(dummy_and_add_feature).tolist()

        data_dict['normal'].to_csv("{}_PCA_{}_TRAIN.csv".format(config['text'], str(config['pca'])), index=False)
        data_dict['abnormal'].to_csv("{}_PCA_{}_TEST.csv".format(config['text'], str(config['pca'])), index=False)
        print("{}_PCA_{}_TRAIN.csv is saved!".format(config['text'], str(config['pca'])))
        print("{}_PCA_{}_TEST.csv is saved!".format(config['text'], str(config['pca'])))
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Preprocessing script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)