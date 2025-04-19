import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

class PerformEDA:
    def __init__(self, data):
        # Убедимся, что столбец 'date' — это datetime
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values(by='date')
    
    def vizualize_var(self, col, windows=[30]):
        window_ts = {}
        for window in windows: 
            window_ts[window] = self.data[col].rolling(window=window).mean()

        fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
        ax.plot(self.data['date'], self.data[col], label='Исходные данные', alpha=0.4)
        for window in windows: 
            ax.plot(self.data['date'], window_ts[window], label=f'Скользящее среднее, window = {window}', linewidth=2)
        ax.set_title(col, fontsize=18)
        ax.set_xlabel('Год', fontsize=14)
        ax.set_ylabel('Значение', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

        # Показываем только года на оси X
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.show()
