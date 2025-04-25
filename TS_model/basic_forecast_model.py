from typing import Optional, Any
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator

class ForecastModel:
    def __init__(self, model: Optional[BaseEstimator] = None):
        """
        Инициализация модели 
        model: модель которая будет использоваться для предсказаний
        """
        self.model = model
        self.trained= False
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, **fit_params: Any) -> None:
        """
        Обучение модели. Это простое обучение без оптимизаций. При желании в fit можно добавить
        любые собственные параметры или валидационную выборку. Как правило может быть необходимо при 
        ручном режиме
        x_train: обучающая выборка
        y_train: целевая переменная
        fit_params: дополнительные параметры для кастомной модели
        """
        self.model.fit(x_train, y_train, **fit_params)
        self.trained = True

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Прогноз с использованием обученной модели
        x_test: тестовая выборка
        """
        return self.model.predict(x_test)
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
        """
        Вычисление MAE для проверки соответствию критерию заказчика
        x_test: тестовая выборка
        y_test: целевая переменная на тестовой выборке (если доступна)
        """
        preds = self.predict(x_test)
        mae = mean_absolute_error(y_test, preds)
        max_err = np.max(np.abs(y_test - preds))
        print(f'MAE = {mae:.3f}, Max Error = {max_err:.3f}')
        if max_err > 0.42:
            print('❌❌ Прогнозы превшают порог ошибки 0.42 ❌❌❌')
        return mae, max_err