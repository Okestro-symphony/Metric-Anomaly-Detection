import os 
import random
import argparse
import numpy as np
import pandas as pd
import pickle as pkl

from tqdm import tqdm
from sklearn.decomposition import PCA
from SYMPHONY_UTILS import scaling, dummy_and_add_feature, trash_col_final, derived_col_list

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Data Preprocessing', add_help=False)

    parser.add_argument('--dataset_path', default='/home/ubuntu/lsy/SYMPHONY/SYMPHONY2/train_data_2.pkl', type=str)
    parser.add_argument('--scaler_path', default='/home/ubuntu/lsy/SYMPHONY/SYMPHONY2/', type=str)
    parser.add_argument('--save_name', default='/home/ubuntu/lsy/SYMPHONY/SYMPHONY2/', type=str)
    parser.add_argument('--n_components', default=24, type=int)
    parser.add_argument('--phase', default='Train', type=str)
    parser.add_argument('--pca', action='store_true')

    return parser


def main(args):
    global trash_col_final
    global derived_col_list
    seed = 10

    config = {
        'dataset_path': args.dataset_path,
        'scaler_path': args.scaler_path,
        'save_name': args.save_name,
        'n_components': args.n_components,
        'phase': args.phase,
        'pca': args.pca,
        }
        
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
        
    # -------------------------------------------------------------------------------------------
        
    config['scaler_path'] = config['scaler_path'] + 'Scaler_{}_PCA_{}_METRIC_PREPROCESSED.pkl'.format(config['n_components'], str(config['pca']))
    config['pca_save_name'] = config['save_name'] + 'PCA_{}_PCA_{}_METRIC_PREPROCESSED.pkl'.format(config['n_components'], str(config['pca']))
    config['save_name'] = config['save_name'] + '{}_{}_PCA_{}_METRIC_PREPROCESSED.csv'.format(config['phase'], config['n_components'], str(config['pca']))

    if config['pca'] == True:
            
        #1. 하나의 DataFrame으로 취합된 데이터셋 로드
        data = pd.read_pickle(config['dataset_path'])
        
        if config['phase'] == 'Train':
                    
            #2. 전체 데이터 중 변수의 값으로 표현되는 비율이 1% 이하일 경우 해당 컬럼 제외
            # Continuos 한지, 안한지를 판단 -> 15% 이하일 경우 해당 컬럼 제외
            drop_col_list = []
            for col in data.columns:
                if (len(data[col].unique()) / len(data)) <= 0.001:
                    drop_col_list.append(col)
            
            # 노드 5,6,7,8로 표현되지 않는 변수 제외
            sustain_col = ['node5_MEM_memtotal', 'node6_MEM_memtotal', 'node7_MEM_memtotal', 'node8_MEM_memtotal', 
                            'app_id', '5_executor_runTime_count', 'Anomaly']

            trash_col_final = list(set(drop_col_list) - set(sustain_col))
        data.drop(trash_col_final, axis=1, inplace=True)
        
        # 3. 전체 변수에 대해 -1값을 NaN값 으로 변환
        data.replace(-1, np.nan, inplace=True)
        data.drop(['3_executor_runTime_count', '4_executor_runTime_count', '5_executor_runTime_count'], axis=1, inplace=True)

        if config['phase'] == 'Train':
            # 4. 전체 데이터 중 변수의 값으로 표현되는 비율이 90% 이상일 경우 해당 컬럼에 대하여 ‘lag’, ‘Rolling_mean’ 등의 파생변수를 생성 
            derived_col_list = []
            for col in data.columns:
                
                if (len(data[col].unique()) / len(data)) >= 0.9:
                    derived_col_list.append(col)
            derived_col_list.append('node5_NET_ib0-read-KB/s')

        lag_col_list = [f'{col}_lag' for col in derived_col_list]  # lag 목록 생성
        Rolling_Mean_col_list = [f'{col}_Rolling_Mean' for col in derived_col_list]  # Rolling_Mean 목록 생성

        data[lag_col_list] = 0 
        data[Rolling_Mean_col_list] = 0 

        # 각각의 인스턴스별로 파생변수 생성
        for instance in tqdm(data.app_id.unique()):  
            df_index = data[data.app_id == instance].index  
            
            data.loc[df_index, lag_col_list] = data[derived_col_list].loc[df_index].shift(1).values  
            data.loc[df_index, Rolling_Mean_col_list] = data[derived_col_list].loc[df_index].rolling(5, min_periods=1).mean().values  

            # 결측값 보간
            data.loc[df_index] = data.loc[df_index].fillna(method='ffill').fillna(method='bfill') 
        
        scale_col = set(data.columns)  # 'bin_df'의 모든 열 이름을 가져옴
        scale_col = list(scale_col - set(['app_id', 'Anomaly']))  # 'host_name'과 'datetime'을 열 이름에서 제외

        data = data.reset_index()
        
        # 5. 각각의 app_id별로 정규화를 수행함
        data_scaled = scaling(config=config, data=data, scale_col=scale_col)  # 'data'에 min-max 스케일링 적용
            
        # 6. PCA 수행
        # 80%는 18개, 85%는 24개, 90%는 39개
        if config['phase'] == 'Train':
            pca = PCA(n_components=config['n_components']) # 주성분을 몇개로 할지 결정
            principalComponents = pca.fit_transform(data_scaled.drop(['t', 'app_id', 'Anomaly'], axis=1))
            pkl.dump(pca, open(config['pca_save_name'],"wb"))
            print("PCA is saved at {}".format(config['pca_save_name']))
            
        else: 
            pca = pkl.load(open(config['pca_save_name'],'rb'))
            principalComponents = pca.transform(data_scaled.drop(['t', 'app_id', 'Anomaly'], axis=1))
            print("PCA is loaded at {}".format(config['pca_save_name']))
            
        principalDf = pd.DataFrame(data=principalComponents)
        PCA_df = pd.concat([principalDf, data_scaled[['t', 'app_id', 'Anomaly']]], axis = 1)

        # 7. 한 주의 요일을 나타내는 'day_of_week' 변수를 생성
        PCA_df['day_of_week'] = PCA_df['t'].dt.dayofweek  # 'day_of_week' 열 추가
        day_of_week_df = pd.get_dummies(PCA_df['day_of_week'], prefix = 'day_of_week', dtype=int)  # 'day_of_week'에 대한 더미 변수 데이터프레임 생성

        PCA_df = pd.concat([PCA_df, day_of_week_df], axis=1)  # 'bin_df'와 'day_of_week_df'를 연결
        PCA_df['t'] = PCA_df['t'].astype(str)  # 'datetime'을 문자열 형식으로 변환

        # Check if 'day_of_week_5' column is present in the DataFrame
        if 'day_of_week_5' not in PCA_df.columns:
            # Add 'day_of_week_5' column with default values (e.g., 0)
            PCA_df['day_of_week_5'] = 0    

        # 8. datetime 변수 변환 -> (Cyclical Encoding)
        # 't'에 'dummy_and_add_feature' 함수 적용 후 반환된 값들에 대해 새로운 열 생성
        PCA_df[['sin_second', 'cos_second', 'sin_minute', 'cos_minute', 'sin_hour', 'cos_hour', 
                    'sin_day', 'cos_day', 'sin_month', 'cos_month']] = PCA_df['t'].apply(dummy_and_add_feature).tolist()  

        PCA_df.drop(['t', 'day_of_week'], axis=1, inplace=True)  # 'datetime'과 'day_of_week' 열 삭제
        PCA_df['Anomaly'] = PCA_df['Anomaly'].apply(lambda x: 1 if x != 0 else 0)
                
        PCA_df.to_csv("{}".format(config['save_name']), index=False)
        print("{} is saved!".format(config['save_name']))
        
    else:

        #1. 하나의 DataFrame으로 취합된 데이터셋 로드
        data = pd.read_pickle(config['dataset_path'])
            
        if config['phase'] == 'Train':
            #2. 전체 데이터 중 변수의 값으로 표현되는 비율이 1% 이하일 경우 해당 컬럼 제외
            # Continuos 한지, 안한지를 판단 -> 15% 이하일 경우 해당 컬럼 제외
            drop_col_list = []
            for col in data.columns:
                if (len(data[col].unique()) / len(data)) <= 0.001:
                    drop_col_list.append(col)
            
            # 노드 5,6,7,8로 표현되지 않는 변수 제외
            sustain_col = ['node5_MEM_memtotal', 'node6_MEM_memtotal', 'node7_MEM_memtotal', 'node8_MEM_memtotal', 
                            'app_id', '5_executor_runTime_count', 'Anomaly']

            trash_col_final = list(set(drop_col_list) - set(sustain_col))
        data.drop(trash_col_final, axis=1, inplace=True)
        
        # 3. 전체 변수에 대해 -1값을 NaN값 으로 변환
        data.replace(-1, np.nan, inplace=True)
        data.drop(['3_executor_runTime_count', '4_executor_runTime_count', '5_executor_runTime_count'], axis=1, inplace=True)

        if config['phase'] == 'Train':
            # 4. 전체 데이터 중 변수의 값으로 표현되는 비율이 90% 이상일 경우 해당 컬럼에 대하여 ‘lag’, ‘Rolling_mean’ 등의 파생변수를 생성 
            derived_col_list = []
            for col in data.columns:
                
                if (len(data[col].unique()) / len(data)) >= 0.9:
                    derived_col_list.append(col)
            derived_col_list.append('node5_NET_ib0-read-KB/s')
        
        lag_col_list = [f'{col}_lag' for col in derived_col_list]  # lag 목록 생성
        Rolling_Mean_col_list = [f'{col}_Rolling_Mean' for col in derived_col_list]  # Rolling_Mean 목록 생성

        data[lag_col_list] = 0 
        data[Rolling_Mean_col_list] = 0 

        # 각각의 고유한 인스턴스별로 파생변수 생성
        for instance in tqdm(data.app_id.unique()):  
            df_index = data[data.app_id == instance].index  
            
            data.loc[df_index, lag_col_list] = data[derived_col_list].loc[df_index].shift(1).values  
            data.loc[df_index, Rolling_Mean_col_list] = data[derived_col_list].loc[df_index].rolling(5, min_periods=1).mean().values  

            # 결측값 보간
            data.loc[df_index] = data.loc[df_index].fillna(method='ffill').fillna(method='bfill') 
        
        scale_col = set(data.columns)  # 'bin_df'의 모든 열 이름을 가져옴
        scale_col = list(scale_col - set(['app_id', 'Anomaly']))  # 'host_name'과 'datetime'을 열 이름에서 제외
        
        data = data.reset_index()
        
        # 5. 각각의 app_id별로 정규화를 수행함
        data_scaled = scaling(config=config, data=data, scale_col=scale_col)  # 'train_data'에 min-max 스케일링 적용
            
        # 6. 한 주의 요일을 나타내는 'day_of_week' 변수를 생성
        data_scaled['day_of_week'] = data_scaled['t'].dt.dayofweek  # 'day_of_week' 열 추가
        day_of_week_df = pd.get_dummies(data_scaled['day_of_week'], prefix = 'day_of_week', dtype=int)  # 'day_of_week'에 대한 더미 변수 데이터프레임 생성

        data_scaled = pd.concat([data_scaled, day_of_week_df], axis=1)  # 'bin_df'와 'day_of_week_df'를 연결
        data_scaled['t'] = data_scaled['t'].astype(str)  # 'datetime'을 문자열 형식으로 변환
        
        # Check if 'day_of_week_5' column is present in the DataFrame
        if 'day_of_week_5' not in data_scaled.columns:
            # Add 'day_of_week_5' column with default values (e.g., 0)
            data_scaled['day_of_week_5'] = 0
        
        # 7. datetime 변수 변환 -> (Cyclical Encoding)
        # 't'에 'dummy_and_add_feature' 함수 적용 후 반환된 값들에 대해 새로운 열 생성
        data_scaled[['sin_second', 'cos_second', 'sin_minute', 'cos_minute', 'sin_hour', 'cos_hour', 
                            'sin_day', 'cos_day', 'sin_month', 'cos_month']] = data_scaled['t'].apply(dummy_and_add_feature).tolist()  

        data_scaled.drop(['t', 'day_of_week'], axis=1, inplace=True)  # 'datetime'과 'day_of_week' 열 삭제
        data_scaled['Anomaly'] = data_scaled['Anomaly'].apply(lambda x: 1 if x != 0 else 0)
        
        data_scaled.to_csv("{}".format(config['save_name']), index=False)
        print("{} is saved!".format(config['save_name']))

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Preprocessing script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)