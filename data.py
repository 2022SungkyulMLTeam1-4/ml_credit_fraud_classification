import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def load_and_preprocess_data(data_path: str) -> (np.ndarray, np.ndarray):
    """
    지정된 경로로부터 데이터를 불러와, 전처리 수행 후 반환합니다.

    :param data_path: csv 데이터 경로
    :returns: 데이터와 레이블의 튜플
    """
    data = pd.read_csv(data_path)
    # Class는 레이블 클래스이며, Time은 단순한 시간 데이터이기에 분류에 불필요한 특성을 추가하는 것으로 판단하였습니다.
    x_data: pd.DataFrame = data.drop(columns=['Class', 'Time'])
    y_data: pd.Series = data['Class']

    x_data, y_data = apply_smote(x_data, y_data)

    return np.array(x_data), np.array(y_data)


def apply_smote(x_data: pd.DataFrame, y_data: pd.Series) -> (pd.DataFrame, pd.Series):
    """
    SMOTE(Synthetic Minority Oversampling TEchnique) 기술을 이용하여 불균형 데이터를 균일하게 만듭니다.\n
    이는 소수 클래스의 인스턴스를 임의로 선택하여, k개의 무작위 소수 클래스 이웃을 찾고, 그 중, 가장 가까운 이웃을 찾아 선분을 잇고,
    선분 사이에서 새로운 합성 데이터를 생성하는 기법입니다.\n
    통상적으로 5를 사용한다고 하기에, 5를 사용하였습니다.

    :param x_data: 레이블을 제외한 모든 데이터
    :param y_data: 레이블 데이터
    :returns: SMOTE로 균일화된 데이터와 레이블을 튜플을 반환합니다.
    """
    oversample = SMOTE(k_neighbors=5)
    x_data, y_data = oversample.fit_resample(x_data, y_data)

    return x_data, y_data
