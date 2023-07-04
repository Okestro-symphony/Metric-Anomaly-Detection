# Metric-Anomaly-Detection

IITP-2021-0-00256 클라우드 자원의 지능적 관리를 위한 이종 가상화(VM+Container) 통합 운용 기술 개발 과제를 통해 공개한 오케스트로의 메트릭 이상탐지 알고리즘

### Directory Explanation
* Anomaly_Detection : 사후 이상탐지 관련 코드 폴더
  > SYMPHONY_MAIN_FINAL.py : 전처리된 데이터 기반 모델 학습 코드
  
  > SYMPHONY_MODEL_FINAL.py : 알고리즘 모델

  > SYMPHONY_PREPROCESSING_FINAL.py : 메트릭 데이터를 입력으로 받아 전처리를 수행

  > SYMPHONY_UTILS_FINAL.py : 데이터 전처리 수행 시 사용되는 함수 및 클래스를 모아둔 코드
  

  
* Real-Time_Anomaly_Detection : 실시간 이상탐지 관련 코드 폴더
  > SYMPHONY_INFERENCE.py : 학습된 모델 바탕 추론 코드

  > SYMPHONY_MODEL.py : 알고리즘 모델

  > SYMPHONY_PREPROCESSING.py : 메트릭 데이터를 입력으로 받아 전처리를 수행

  > SYMPHONY_TRAIN.py : 전처리된 데이터 기반 모델 학습 코드

  > SYMPHONY_UTILS.py : 데이터 전처리 수행 시 사용되는 함수 및 클래스를 모아둔 코드


### Who We Are
회사 홈페이지:
http://okestro.com/

### License
MIT License
