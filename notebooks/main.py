import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import joblib, os
import schedule
import time

sys.path.append('../')

from TS_model.vizualization import PerformEDA
from TS_model.feature_engeneering import FeatureEngineer
from TS_model.automl_tuning import ModelSelector
from TS_model.fallback_train import FallbackModel
from TS_model.breakpoints_detector import ChangeFinder
from TS_model.anomaly_detector import AnomalyDetector, RealTimeAnomalyDetector


def main():
    """
    Логика:
    0. Прогон и работа модели каждый день в 6 вечера 
    P.S. нами был создан класс для парсинга новых данных - Class_date.py,
    но т.к. новых данных нет в доступе, ниже в реализации подгружаем исторические данные по фичам
    1. делаем предсказание сальдо и выводим абсолютную ошибку прогноза модели, записываем результат в predictions
    2. используем Isolation Forest для обнаружения аномалий на историческом датасете и его калибровки
    3. анализируем на наличие concept drift и является ли новое значение аномалией
    (если получтли где-то True -> запускаем переобучение/переводим модель в ручное управление)

    """
    MODEL_PATH = 'model.pkl'
    TODAY      = pd.Timestamp.today().tz_localize(None)
    #Если первый день месяца, то переобучаем модельку
    FIRST_DAY  = TODAY.day == 1

    data = pd.read_excel('../Data/chosen_features.xlsx', index_col=0)
    x, y = data.drop('balance', axis=1), data['balance']
    # предскажем последнее значение и выведем для него MAE 
    x_train, y_train, x_test, y_test = x.iloc[0:-1], y.iloc[0:-1], x.iloc[[-1]], y.iloc[[-1]]

    #используем уже обученный AutoML
    x_train.insert(0, 'date', x_train.index)
    model = joblib.load(MODEL_PATH)
    print('Загрузили существующую модель')
    x_test.insert(0, 'date', x_test.index)
    prediction = model.predict(x_test)
    prediction.to_excel('../Data/preds.xlsx', index = False)

    #рассчитаем абсолютную ошибку полученного предсказания  
    abs_error = abs(y_test - prediction)
    print(f'абсолютная ошибка полученного предсказания: {abs_error:.3f}')

    # проверяем на concept shift & аномалии
    # объявим детекторы
    cm = ChangeFinder(alpha=0.05, beta=0.005, method='cusum', mean_diff=-0.01, 
                              trsh=0.06, slice_length=5, breaks_max=3)
    cr = ChangeFinder(alpha=0.01, beta=0.07, method='sr', sigma_diff = 0.5, ceil=100,
                              trsh=10, slice_length=10, breaks_max=2)
    
    # применение Isolation Forest для обнаружения аномалий
    # Обучаем базовый детектор на исторических данных 
    historical_data = data.iloc[0:-1]
    _, _, anomaly_detector_iforest = AnomalyDetector.process(historical_data)
    # Создаем real-time детектор
    rt_detector = RealTimeAnomalyDetector(anomaly_detector_iforest)
    new_features = x_test

    if cm.detect_anomaly_real_time(prediction) or cr.detect_anomaly_real_time(prediction):
        trigger_1 = True  # переобучение, если concept_drift
    elif rt_detector.predict_new_value(new_features, prediction):
        trigger_2 = True # перевод в ручное управление, если аномалии
    else:
        trigger_1 = False
        trigger_2 = False

    if trigger_2:
        fb_model = FallbackModel()
        model = fb_model.train()
        print('Используем пользовательскую модель (fallback)')
    elif trigger_1 or FIRST_DAY or not os.path.exists(MODEL_PATH):
        # нужно переобучить AutoML и сохранить - time_budget(время обучения)
        x_train.insert(0, 'date', x_train.index)
        model = ModelSelector(time_budget=300 * 12, metric='mae').find_best_model(x_train, y_train, period=1)
        joblib.dump(model, MODEL_PATH)
        print('Переобучили AutoML')
    else:
        None

    pass

schedule.every().day.at("18:00").do(main)

if __name__ == 'main':
    while True:
        schedule.run_pending()
        time.sleep(60)