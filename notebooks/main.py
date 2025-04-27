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
from TS_model.breakpoints_detector import UniversalChangePointDetector
from TS_model.automl_tuning import ModelSelector


def main():
    # Логика в мейн следующая:
    # 0. Прогон и работа модели каждый день в 6 вечера 
    # 1. Загружаем вчерашний прогноз (файл predictions_data.xlsx) и сегодняшнее значение(по api или psrser)и выводим ошибку прогноза
    # 2. Загружаем данные которые лежат в features_data.xlsx и добавляем к ним сегодняшние знаяения 
    # 3. anomaly detection - присвоили флаг (на основе features_data) и вывели визуализацию
    # 4. concept drift - присвоили флаг (на основе features_data) и вывели графики
    # 5  Если сработали флаги то берем features_data (без сегоднящних данных) и делаем feature slection
    # 6. В зависимости от того какие данные сработали идем в разные сценарии auto_ml
    # 7. Деламе предсказание, пересохраняем features_data с новыми данными, в predictions_data добавляем предскзаание на завтра, а вчеращнее предсказание перезатираем на актуальные данные
    pass


schedule.every().day.at("18:00").do(main)

if __name__ == 'main':
    while True:
        schedule.run_pending()
        time.sleep(60)