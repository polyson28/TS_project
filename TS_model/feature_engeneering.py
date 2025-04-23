import pandas as pd
import numpy as np
from typing import List, Union

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, date_col: str = 'date'):
        """
        Инициализация данных, содержащих как минимум следующие столбцы:
        date_col, 'inflow', 'outflow', 'balance', 
        а также любые существующие макро признаки
        """
        self.df = df.copy()
        self.date_col = date_col
        self.df.set_index(date_col, inplace=True)
        self.features = pd.DataFrame(index=self.df.index)
    
    def add_lag_features(self, lags: List[int] = [1, 2, 7]) -> 'FeatureEngineer':
        """
        Добавление признаков лагов для указанных периодов лагов
        """
        for lag in lags:
            self.features[f'balance_lag{lag}'] = self.df['balance'].shift(lag)
            self.features[f'inflow_lag{lag}'] = self.df['inflow'].shift(lag)
            self.features[f'outflow_lag{lag}'] = self.df['outflow'].shift(lag)
        return self
    
    def add_rolling_features(self, windows: List[int] = [3, 7, 30]) -> 'FeatureEngineer':
        """
        Добавление скользящего окна для указанных размеров окон (в днях)
        """
        for w in windows:
            # shift(1) для того чтобы текущий элемент не входил в окно
            self.features[f'balance_ma{w}'] = self.df['balance'].shift(1).rolling(window=w, min_periods=1).mean()
            self.features[f'inflow_ma{w}'] = self.df['inflow'].shift(1).rolling(window=w, min_periods=1).mean()
            self.features[f'outflow_ma{w}'] = self.df['outflow'].shift(1).rolling(window=w, min_periods=1).mean()
        return self
    
    def add_seasonal_features(self) -> 'FeatureEngineer':
        """
        Добавление сезонных индикаторов, таких как день недели и месяц
        """
        self.features['day_of_week'] = self.features.index.dayofweek
        self.features['month'] = self.features.index.month
        self.features['day_of_week_sin'] = np.sin(2 * np.pi * self.features['day_of_week'] / 7)
        self.features['day_of_week_cos'] = np.cos(2 * np.pi * self.features['day_of_week'] / 7)
        self.features['month_sin'] = np.sin(2 * np.pi * self.features['month'] / 12)
        self.features['month_cos'] = np.cos(2 * np.pi * self.features['month'] / 12)

        self.features = pd.get_dummies(self.features, columns=['day_of_week'], prefix='dow', drop_first=False)
        return self
    
    def add_special_dates(self, tax_dates: Union[set, List[Union[pd.Timestamp, str]]]) -> 'FeatureEngineer':
        """
        Добавление бинарного признака из налогового календаря
        """
        self.features['tax_day'] = 0
        tax_dates = pd.to_datetime(tax_dates)
        self.features.loc[self.features.index.isin(tax_dates), 'tax_day'] = 1
        return self
    
    def add_macro_features(self, macro_df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Добавление макро переменных (уже выровненных по дате)
        """
        self.features = self.features.join(macro_df, how='left')
        # forward-fill для заполнения последнего известного значения, если данные не ежедневные
        self.features.fillna(method='ffill', inplace=True)
        return self
    
    def get_feature_df(self) -> pd.DataFrame:
        """
        Возврат конечного датасета признаков 
        """
        # Удаление строк с NaN (из-за лагов) в начале
        feature_df = self.features.copy()
        feature_df.drpona(inplace=True)

        return feature_df