import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma
import matplotlib.pyplot as plt
from typing import Union, List
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV


class WrapperMethod:
    def __init__(self):
        """Класс для реализации оберточных методов для feature selection 
        """
        pass
    
    def fit(self, features: pd.DataFrame, target: Union[pd.DataFrame, pd.Series]):
        """фит данных 

        Args:
            features (pd.DataFrame): датасет с признаками 
            target (Union[pd.DataFrame, pd.Series]): датасет с таргетом 
        """
        self.features = features
        self.target = target 
    
    def implement(self, model: Union[LassoCV, RidgeCV, ElasticNetCV, RandomForestRegressor], tscv: TimeSeriesSplit) -> List[str]:
        """реализация отбора признаков с помощью оберточных методов 

        Args:
            model (Union[LassoCV, RidgeCV, ElasticNetCV, RandomForestRegressor]): используемая модель 
            tscv (TimeSeriesSplit): time series split для кросс валидации 

        Raises:
            ValueError: если метод вызван до fit()

        Returns:
            List[str]: список с важными фичами 
        """
        if not hasattr(self, 'features') or not hasattr(self, 'target'):
            raise ValueError("Сначала нужно зафитить данные")
        
        selector = RFECV(
            estimator=model,
            step=1,
            cv=tscv,
            scoring='neg_mean_squared_error',
            min_features_to_select=5,
            n_jobs=-1
        )

        selector.fit(self.features, self.target)

        # Вывод результатов
        selected_features = self.features.columns[selector.support_]
        return selected_features


class TransferEntropyFeatureSelection:
    def __init__(self, k: int, delay: int=1, embed_dim: int=1):
        """Класс для проведения feature selecton с помощью трансфертной энтропии 
        
        Args:
            k (int): количество ближайших соседей 
            delay (int, optional): временной лаг для вложения. Defaults to 1.
            embed_dim (int, optional): размерность вложения (количество предыдущих точек, используемых для реконструкции пространства состояний). Defaults to 1.
        """
        self.k = k
        self.delay = delay
        self.embed_dim = embed_dim
        self.feature_importances_ = None
        
    def compute_border_points(self, X: np.ndarray, indices: np.ndarray, k: int) -> Union[np.ndarray, np.ndarray]:
        """Вычисление характеристик для оценки локальных объемов и граничных точек 
        (для каждой точки расчет объема минимального гиперпрямоугольника, охватывающего всех соседей, и расчет количества соседей на границе)

        Args:
            X (np.ndarray): данные с признаками
            indices (np.ndarray): индексы k ближайших соседей для каждой точки (без самой точки)
            k (int): количество ближайших соседей 

        Returns:
            Union[np.ndarray, np.ndarray]: массив border_points с логарифмами объемов минимальных гиперпрямоугольников, 
                массив log_volumes с количеством соседей, лежащих на границе прямоугольника 
        """
        log_volumes   = np.zeros(X.shape[0])
        border_points = np.zeros(X.shape[0], dtype=int)

        for i in range(X.shape[0]):
            nbrs   = X[indices[i]]               
            diffs  = np.abs(nbrs - X[i])          
            # max_d  = diffs.max(axis=0)          

            # # объем минимального гиперпрямоугольника
            # log_volumes[i] = np.sum(np.log(2 * max_d))
            max_d = diffs.max(axis=0)
            max_d[max_d <= 0] = np.finfo(float).eps  
            log_volumes[i] = np.sum(np.log(2 * max_d))

            # считаем, на скольких соседях хотя бы в одной оси достигается max_d
            is_border = np.zeros(k, dtype=bool)
            for ax in range(X.shape[1]):
                mask = np.isclose(diffs[:, ax], max_d[ax], atol=1e-12)
                is_border |= mask
            border_points[i] = is_border.sum()

        return border_points, log_volumes
    
    def estimate_entropy(self, X: np.ndarray, k: int) -> float:
        tree = cKDTree(X)
        dists, inds = tree.query(X, k=k+1, p=np.inf)
        inds = inds[:, 1:]
        
        border_points, log_vol = self.compute_border_points(X, inds, k)
        
        H = np.mean(-log_vol + np.log(k) - np.log(X.shape[0]-1) 
                    + digamma(border_points + 1))  # Добавлен член с digamma
        return H
    
    def estimate_transfer_entropy(self, X: np.ndarray, Y: np.ndarray, k: int, delay: int=1, embed_dim: int=1) -> float:
        """Оценка трансфертной энтропии от Y к X 

        Args:
            X (np.ndarray): временной ряд 
            Y (np.ndarray): временной ряд 
            k (int): количество ближайших соседей 
            delay (int, optional): временной лаг для вложения. Defaults to 1.
            embed_dim (int, optional): размерность вложения (количество предыдущих точек, используемых для реконструкции пространства состояний). Defaults to 1.

        Returns:
            float: оценка трансфертной энтропии от Y к X
        """
        T = len(X)
        # число «доступных» точек после вложения
        N = T - embed_dim*delay - 1

        # будущее X_{t+1}
        X_fut = X[embed_dim*delay + 1 :].reshape(-1, 1)  # shape (N,1)

        # прошлое X и Y
        X_p = np.zeros((N, embed_dim))
        Y_p = np.zeros((N, embed_dim))
        for i in range(embed_dim):
            start = embed_dim*delay - i*delay
            end   = -(i*delay + 1)
            X_p[:, i] = X[start : end]
            Y_p[:, i] = Y[start : end]

        # вычисляем четыре энтропии
        H_xp_x = self.estimate_entropy(np.hstack([X_fut, X_p]), k)
        H_x = self.estimate_entropy(X_p, k)
        H_xp_xy = self.estimate_entropy(np.hstack([X_fut, X_p, Y_p]), k)
        H_xy = self.estimate_entropy(np.hstack([X_p,   Y_p]), k)

        te = H_xp_x - H_x - H_xp_xy + H_xy
        return max(0.0, te)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """фит на данных, для каждого признака вычисляем TE от него к y

        Args:
            X (np.ndarray): признаки 
            y (np.ndarray): таргет
        """
        n, d = X.shape
        imp = np.zeros(d)
        for j in range(d):
            imp[j] = self.estimate_transfer_entropy(y, X[:, j],
                k=self.k,
                delay=self.delay,
                embed_dim=self.embed_dim)
        # нормируем (если не все нули)
        s = imp.sum()
        if s > 0:
            imp /= s
        self.feature_importances_ = imp
        return self

    def transform(self, X: np.ndarray, threshold: float=0.01) -> np.ndarray:
        """удаление признаков с важностью ниже threshold

        Args:
            X (np.ndarray): признаки
            threshold (float, optional): порог, начиная с которого отсеиваем признаки. Defaults to 0.01.

        Raises:
            ValueError: если метод был вызван до фита на данных

        Returns:
            np.ndarray: важные признаки
        """
        if self.feature_importances_ is None:
            raise ValueError("Сначала нужно зафитить данные")
        mask = self.feature_importances_ >= threshold
        return X[:, mask]

    def selected_features(self, threshold: float=0.01) -> np.ndarray:
        """удаление признаков с важностью ниже threshold

        Args:
            threshold (float, optional): порог, начиная с которого отсеиваем признаки. Defaults to 0.01.

        Raises:
            ValueError: если метод был вызван до фита на данных

        Returns:
            np.ndarray: индексы с важными признаками
        """
        if self.feature_importances_ is None:
            raise ValueError("Сначала нужно зафитить данные")
        return np.where(self.feature_importances_ >= threshold)[0]
    
    def selected_feature_names(self,
        feature_names: List[str],
        threshold: float=0.01
    ) -> List[str]:
        """
        Имена важных признаков среди переданного списка feature_names.

        Args:
            feature_names (List[str]): список с названиями столбцов
            threshold (float, optional): порог, начиная с которого отсеиваем признаки. Defaults to 0.01.

        Returns:
            list[str]: имена признаков, чья важность ≥ threshold.

        Raises:
            ValueError: если метод вызван до fit() или длина feature_names
                не совпадает с числом рассчитанных importances.
        """
        if self.feature_importances_ is None:
            raise ValueError("Сначала нужно зафитить данные")

        idx = self.selected_features(threshold)
        return [feature_names[i] for i in idx]