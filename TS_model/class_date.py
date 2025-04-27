import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date

class MarketDataFetcher:
    def __init__(self):
        self.imoex_url = "https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities/IMOEX.json"
        self.imoex_data = None
        self.usd_rub_data = None
        self.inflation_data = None
        self.trade_balance_data = None
        self.unemployment_data = None
        self.gdp_per_capita_data = None
        self.consumer_expenses_data = None
        self.age_structure_data = None
        self.migration_balance_data = None
        self.digitalization_level_data = None
        self.calendar_df = None
        self.proj_file = None
        self.population_density_data = None


    # --- Блок получения основных данных ---
    def fetch_imoex(self, start_date='2017-01-01', end_date='2021-12-31'):
    # Проверка на то, если start_date и end_date одинаковы, то будем загружать только данные за одну дату
      if start_date == end_date:
        params = {
            'from': start_date,
            'till': end_date,
            'start': 0
        }
        all_records = []

        response = requests.get(self.imoex_url, params=params)
        data = response.json()

        columns = data['history']['columns']
        rows = data['history']['data']

        if rows:
            df = pd.DataFrame(rows, columns=columns)
            all_records.append(df)

        imoex_df = pd.concat(all_records, ignore_index=True)
        imoex_df = imoex_df[['TRADEDATE', 'CLOSE']].rename(columns={'TRADEDATE': 'date'})
        imoex_df['date'] = pd.to_datetime(imoex_df['date'])
        imoex_df.set_index('date', inplace=True)

        imoex_df.reset_index(inplace=True)
        imoex_df['date'] = imoex_df['date'].dt.strftime('%Y-%m-%d')

        self.imoex_data = imoex_df
        return imoex_df

      else:
        # Для временного промежутка
        params = {
            'from': start_date,
            'till': end_date,
            'start': 0
        }
        all_records = []

        while True:
            response = requests.get(self.imoex_url, params=params)
            data = response.json()

            columns = data['history']['columns']
            rows = data['history']['data']

            if not rows:
                break

            df = pd.DataFrame(rows, columns=columns)
            all_records.append(df)

            params['start'] += 100

        imoex_df = pd.concat(all_records, ignore_index=True)
        imoex_df = imoex_df[['TRADEDATE', 'CLOSE']].rename(columns={'TRADEDATE': 'date'})
        imoex_df['date'] = pd.to_datetime(imoex_df['date'])

        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        imoex_df.set_index('date', inplace=True)
        full_df = pd.DataFrame(index=all_dates).join(imoex_df, how='left')
        full_df['CLOSE'].fillna(0, inplace=True)

        full_df.reset_index(inplace=True)
        full_df.rename(columns={'index': 'date'}, inplace=True)
        full_df['date'] = full_df['date'].dt.strftime('%Y-%m-%d')

        self.imoex_data = full_df
        return full_df


    def fetch_usd_rub(self, start_date='2017-01-01', end_date='2021-12-31'):
      if start_date == end_date:
        # Если даты одинаковы, то получаем данные за один день
        data = yf.download("USDRUB=X", start=start_date, end=(pd.to_datetime(start_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
      else:
        # Если даты разные, то получаем данные за промежуток времени
        data = yf.download("USDRUB=X", start=start_date, end=end_date)

      df = data[['Close']].reset_index()
      df.columns = ['date', 'USD_Rate']
      df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

      self.usd_rub_data = df
      return df

    # --- Блок инфляции ---


    def get_inflation_rate(self, year=None, date=None):
       if date:
          # Если передана конкретная дата
          url = f"http://api.worldbank.org/v2/country/RU/indicator/FP.CPI.TOTL.ZG?date={date.year}&format=json"
          response = requests.get(url)
          if response.status_code == 200:
             data = response.json()
             for entry in data[1]:
                if entry['date'] == str(date.year):
                    return entry['value']
          return None
       elif year:
          # Если передан год
          url = f"http://api.worldbank.org/v2/country/RU/indicator/FP.CPI.TOTL.ZG?date={year}&format=json"
          response = requests.get(url)
          if response.status_code == 200:
             data = response.json()
             for entry in data[1]:
                if entry['date'] == str(year):
                    return entry['value']
          return None
       return None

    def generate_daily_inflation_rate(self, start_date=None, end_date=None, year=None):
       all_data = []

       if start_date and end_date:
           # Если передан диапазон дат
           current_date = start_date
           while current_date <= end_date:
                inflation_rate = self.get_inflation_rate(date=current_date)
                if inflation_rate is not None:
                   all_data.append({
                     'date': current_date.strftime('%Y-%m-%d'),
                     'inflation_rate': inflation_rate
                 })
                current_date += timedelta(days=1)

       elif year:
         # Если передан только год
           inflation_rate = self.get_inflation_rate(year=year)
           if inflation_rate is not None:
             is_leap_year = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
             days_in_year = 366 if is_leap_year else 365

             start_date = datetime(year, 1, 1)
             for day in range(days_in_year):
                 date = start_date + timedelta(days=day)
                 all_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'inflation_rate': inflation_rate
                })

       df = pd.DataFrame(all_data)
       self.inflation_data = df
       return df


    # --- Блок торгового баланса ---
    def get_trade_balance(self, year=None, date=None):
     if date:
        # Если передана конкретная дата
        export_url = f"http://api.worldbank.org/v2/country/RU/indicator/NE.EXP.GNFS.CD?date={date.year}&format=json"
        import_url = f"http://api.worldbank.org/v2/country/RU/indicator/NE.IMP.GNFS.CD?date={date.year}&format=json"

        export_response = requests.get(export_url)
        import_response = requests.get(import_url)

        if export_response.status_code == 200 and import_response.status_code == 200:
            export_data = export_response.json()
            import_data = import_response.json()

            export_value = None
            import_value = None

            for entry in export_data[1]:
                if entry['date'] == str(date.year):
                    export_value = entry['value']
            for entry in import_data[1]:
                if entry['date'] == str(date.year):
                    import_value = entry['value']

            if export_value is not None and import_value is not None:
                return export_value - import_value

        return None
     elif year:
        # Если передан год
        export_url = f"http://api.worldbank.org/v2/country/RU/indicator/NE.EXP.GNFS.CD?date={year}&format=json"
        import_url = f"http://api.worldbank.org/v2/country/RU/indicator/NE.IMP.GNFS.CD?date={year}&format=json"

        export_response = requests.get(export_url)
        import_response = requests.get(import_url)

        if export_response.status_code == 200 and import_response.status_code == 200:
            export_data = export_response.json()
            import_data = import_response.json()

            export_value = None
            import_value = None

            for entry in export_data[1]:
                if entry['date'] == str(year):
                    export_value = entry['value']
            for entry in import_data[1]:
                if entry['date'] == str(year):
                    import_value = entry['value']

            if export_value is not None and import_value is not None:
                return export_value - import_value

        return None
     return None

    def generate_daily_trade_balance(self, start_date=None, end_date=None, year=None):
      all_data = []

      if start_date and end_date:
        # Если передан диапазон дат
        current_date = start_date
        while current_date <= end_date:
            trade_balance = self.get_trade_balance(date=current_date)
            if trade_balance is not None:
                all_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'trade_balance': trade_balance
                })
            current_date += timedelta(days=1)

      elif year:
        # Если передан только год
        trade_balance = self.get_trade_balance(year=year)
        if trade_balance is not None:
            is_leap_year = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
            days_in_year = 366 if is_leap_year else 365

            start_date = datetime(year, 1, 1)
            for day in range(days_in_year):
                date = start_date + timedelta(days=day)
                all_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'trade_balance': trade_balance
                })

      df = pd.DataFrame(all_data)
      self.trade_balance_data = df
      return df

    # --- Блок безработицы ---
    def get_unemployment_rate(self, year=None, date=None):
      if date:
        # Если передана конкретная дата
        url = f"http://api.worldbank.org/v2/country/RU/indicator/SL.UEM.TOTL.ZS?date={date.year}&format=json"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            for entry in data[1]:
                if entry['date'] == str(date.year):
                    return entry['value']

        return None
      elif year:
        # Если передан год
        url = f"http://api.worldbank.org/v2/country/RU/indicator/SL.UEM.TOTL.ZS?date={year}&format=json"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            for entry in data[1]:
                if entry['date'] == str(year):
                    return entry['value']

        return None
      return None

    def generate_daily_unemployment_rate(self, start_date=None, end_date=None, year=None):
       all_data = []

       if start_date and end_date:
        # Если передан диапазон дат
        current_date = start_date
        while current_date <= end_date:
            unemployment_rate = self.get_unemployment_rate(date=current_date)
            if unemployment_rate is not None:
                all_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'unemployment_rate': unemployment_rate
                })
            current_date += timedelta(days=1)

       elif year:
        # Если передан только год
        unemployment_rate = self.get_unemployment_rate(year=year)
        if unemployment_rate is not None:
            is_leap_year = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
            days_in_year = 366 if is_leap_year else 365

            start_date = datetime(year, 1, 1)
            for day in range(days_in_year):
                date = start_date + timedelta(days=day)
                all_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'unemployment_rate': unemployment_rate
                })

       df = pd.DataFrame(all_data)
       self.unemployment_data = df
       return df

    # --- Блок GDP на душу населения ---
    def get_gdp_per_capita(self, start_date=None, end_date=None, year=None, date=None):
     if date:
        # Если передана конкретная дата
        url = f"http://api.worldbank.org/v2/country/RU/indicator/NY.GDP.PCAP.CD?date={date.year}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for entry in data[1]:
                if entry['date'] == str(date.year):
                    return entry['value']
        return None
     elif year:
        # Если передан только год
        url = f"http://api.worldbank.org/v2/country/RU/indicator/NY.GDP.PCAP.CD?date={year}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for entry in data[1]:
                if entry['date'] == str(year):
                    return entry['value']
        return None
     return None

    def generate_daily_gdp_per_capita(self, start_date=None, end_date=None, year=None, date=None):
     all_data = []

     if start_date and end_date:
        # Если передан диапазон дат
        current_date = start_date
        while current_date <= end_date:
            gdp_per_capita = self.get_gdp_per_capita(date=current_date)
            if gdp_per_capita is not None:
                all_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'gdp_per_capita': gdp_per_capita
                })
            current_date += timedelta(days=1)

     elif year:
        # Если передан только год
        gdp_per_capita = self.get_gdp_per_capita(year=year)
        if gdp_per_capita is not None:
            is_leap_year = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
            days_in_year = 366 if is_leap_year else 365

            start_date = datetime(year, 1, 1)
            for day in range(days_in_year):
                date = start_date + timedelta(days=day)
                all_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'gdp_per_capita': gdp_per_capita
                })

     df = pd.DataFrame(all_data)
     self.gdp_per_capita_data = df
     return df

    # --- Блок потребительских расходов ----------------------------------------------------------------------------------------------------------------------------------------------------------------


    def get_consumer_confidence_index(self, start_date=None, end_date=None, year=None, date=None):
     if date:
        # Для конкретной даты
        url = f"http://api.worldbank.org/v2/country/RU/indicator/NE.CON.TOTL.ZS?date={date.year}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for record in data[1]:
                if record['date'] == str(date.year):
                    return record['value']
        return None
     elif start_date and end_date:
        # Для диапазона дат
        url = f"http://api.worldbank.org/v2/country/RU/indicator/NE.CON.TOTL.ZS?date={start_date.year}:{end_date.year}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            result = []

            for record in data[1]:
                year = str(record['date'])
                value = record['value']

                start_date = datetime.strptime(year, "%Y")
                days_in_year = 366 if (start_date.year % 4 == 0 and (start_date.year % 100 != 0 or start_date.year % 400 == 0)) else 365

                for day in range(days_in_year):
                    date = start_date + timedelta(days=day)
                    result.append([date.strftime("%Y-%m-%d"), value])

            df = pd.DataFrame(result, columns=['date', 'consumer_confidence_index'])
            self.consumer_expenses_data = df
            return df
     elif year:
          # Для одного года
        url = f"http://api.worldbank.org/v2/country/RU/indicator/NE.CON.TOTL.ZS?date={year}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and len(data[1]) > 0:
                result = []

                for record in data[1]:
                    if record['date'] == str(year):
                        value = record['value']
                        start_date = datetime.strptime(str(year), "%Y")
                        days_in_year = 366 if (start_date.year % 4 == 0 and (start_date.year % 100 != 0 or start_date.year % 400 == 0)) else 365

                        for day in range(days_in_year):
                            date = start_date + timedelta(days=day)
                            result.append([date.strftime("%Y-%m-%d"), value])

                df = pd.DataFrame(result, columns=['date', 'consumer_confidence_index'])
                self.consumer_expenses_data = df
                return df

     return None

     # Популяция
    def get_population_density(self, date_or_year):
        if isinstance(date_or_year, datetime):  # Если передана конкретная дата
            year = date_or_year.year
        else:  # Если передан год
            year = date_or_year


        url = f"http://api.worldbank.org/v2/country/RU/indicator/EN.POP.DNST?date={year}&format=json"
        print(url)
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(response.json())
            for entry in data[1]:
                if entry['date'] == str(year):
                    return entry['value']
        return None

    def generate_population_density(self, start_date_or_year, end_date_or_year=None):
        all_data = []

        if isinstance(start_date_or_year, datetime):  # Если передана конкретная дата начала
            start_date = start_date_or_year
            end_date = start_date_or_year  # Период на одну дату
        else:  # Если передан диапазон лет
            start_date = datetime(start_date_or_year, 1, 1)
            end_date = datetime(end_date_or_year, 12, 31)

        # Получаем данные для каждой даты или для всех лет
        current_date = start_date
        while current_date <= end_date:
            population_density = self.get_population_density(current_date)
            if population_density is not None:
                all_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'population_density': population_density
                })
            current_date += timedelta(days=1)

        df = pd.DataFrame(all_data)
        self.population_density_data = df
        return df




    # --- Блок возрастной структуры ---
    def get_age_structure(self, date_or_year):
        """
        Получить возрастную структуру для конкретного года или даты.
        Если передана дата, извлекается год.
        """
        if isinstance(date_or_year, datetime):  # Если передана конкретная дата
            year = date_or_year.year
        else:  # Если передан год
            year = date_or_year

        indicators = {
            '0-14': 'SP.POP.0014.TO.ZS',  # Доля населения 0-14 лет
            '15-64': 'SP.POP.1564.TO.ZS',  # Доля населения 15-64 лет
            '65+': 'SP.POP.65UP.TO.ZS'     # Доля населения 65+
        }
        data = {}
        for age_group, indicator in indicators.items():
            url = f"http://api.worldbank.org/v2/country/RU/indicator/{indicator}?date={year}&format=json"
            response = requests.get(url)
            if response.status_code == 200:
                result = response.json()
                for entry in result[1]:
                    if entry['date'] == str(year):
                        data[age_group] = entry['value']
            else:
                data[age_group] = None
        return data

    def generate_age_structure(self, start_date_or_year, end_date_or_year=None):

        all_data = []

        # Если передана конкретная дата, создаем период из одной даты
        if isinstance(start_date_or_year, datetime):
            start_date = start_date_or_year
            end_date = start_date_or_year  # Период на одну дату
        else:  # Если передан диапазон лет
            start_date = datetime(start_date_or_year, 1, 1)
            end_date = datetime(end_date_or_year, 12, 31)

        # Получаем данные для каждой даты или для всех лет
        current_date = start_date
        while current_date <= end_date:
            age_structure = self.get_age_structure(current_date)
            if age_structure:
                all_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'age_0_14': age_structure.get('0-14'),
                    'age_15_64': age_structure.get('15-64'),
                    'age_65_plus': age_structure.get('65+')
                })
            current_date += timedelta(days=1)

        # Конвертируем данные в DataFrame
        df = pd.DataFrame(all_data)
        self.age_structure_data = df
        return df


        #Миграционный баланс
    def get_migration_balance(self, date_or_year):
        """
        Получить миграционный баланс для конкретного года или даты.
        Если передана дата, извлекается год.
        """
        if isinstance(date_or_year, datetime):  # Если передана конкретная дата
            year = date_or_year.year
        else:  # Если передан год
            year = date_or_year

        url = f"http://api.worldbank.org/v2/country/RU/indicator/SM.POP.NETM?date={year}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for entry in data[1]:
                if entry['date'] == str(year):
                    return entry['value']
        return None

    def generate_migration_balance(self, start_date_or_year, end_date_or_year=None, specific_date=None):
        """
        Генерирует данные миграционного баланса для одного дня или для всего периода.
        Если передана дата, возвращаются данные для этой даты.
        Если передан диапазон, возвращаются данные для каждого дня в периоде.
        """
        all_data = []

        # Если передана конкретная дата, создаем период из одной даты
        if isinstance(start_date_or_year, datetime):
            start_date = start_date_or_year
            end_date = start_date_or_year  # Период на одну дату
        else:  # Если передан диапазон лет
            start_date = datetime(start_date_or_year, 1, 1)
            end_date = datetime(end_date_or_year, 12, 31)

        # Получаем данные для каждой даты или для всех лет
        current_date = start_date
        while current_date <= end_date:
            migration_balance = self.get_migration_balance(current_date)
            if migration_balance is not None:
                all_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'migration_balance': migration_balance
                })
            current_date += timedelta(days=1)

        # Конвертируем данные в DataFrame
        df = pd.DataFrame(all_data)

        # Если указан конкретный день, фильтруем данные по этой дате
        if specific_date:
            specific_date_str = specific_date.strftime('%Y-%m-%d')
            df = df[df['date'] == specific_date_str]

        self.migration_balance_data = df
        return df



    # --- Блок цифровизации ---
    def get_digitalization_level(self, date_or_year):
        """
        Получить уровень цифровизации для конкретного года или даты.
        Если передана дата, извлекается год.
        """
        if isinstance(date_or_year, datetime):  # Если передана конкретная дата
            year = date_or_year.year
        else:  # Если передан год
            year = date_or_year

        url = f"http://api.worldbank.org/v2/country/RU/indicator/IT.NET.USER.ZS?date={year}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for entry in data[1]:
                if entry['date'] == str(year):
                    return entry['value']
        return None

    def generate_digitalization_level(self, start_date_or_year, end_date_or_year=None, specific_date=None):
        """
        Генерирует данные уровня цифровизации для одного дня или для всего периода.
        Если передана дата, возвращаются данные для этой даты.
        Если передан диапазон, возвращаются данные для каждого дня в периоде.
        """
        all_data = []

        # Если передана конкретная дата, создаем период из одной даты
        if isinstance(start_date_or_year, datetime):
            start_date = start_date_or_year
            end_date = start_date_or_year  # Период на одну дату
        else:  # Если передан диапазон лет
            start_date = datetime(start_date_or_year, 1, 1)
            end_date = datetime(end_date_or_year, 12, 31)

        # Получаем данные для каждой даты или для всех лет
        current_date = start_date
        while current_date <= end_date:
            digitalization_level = self.get_digitalization_level(current_date)
            if digitalization_level is not None:
                all_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'digitalization_level': digitalization_level
                })
            current_date += timedelta(days=1)

        # Конвертируем данные в DataFrame
        df = pd.DataFrame(all_data)

        # Если указан конкретный день, фильтруем данные по этой дате
        if specific_date:
            specific_date_str = specific_date.strftime('%Y-%m-%d')
            df = df[df['date'] == specific_date_str]

        self.digitalization_level_data = df
        return df

    # --- Блок календарь + налоги ---
    START_YEAR = 2017
    END_YEAR = 2025

    TAX_TYPES = [
        "НДС", "НДФЛ", "Страховые взносы",
        "Акцизы", "Налог на имущество организаций",
        "Транспортный налог", "Налог на землю",
        "Налог на прибыль"
    ]

    TAX_EVENTS = {
        "2017-01-25": ["НДФЛ", "НДС"],
        "2017-02-15": ["Страховые взносы"],
        "2017-04-28": ["НДФЛ", "НДС"],
        "2017-05-01": ["Акцизы"],
        "2017-10-10": ["Налог на имущество организаций"],
        "2018-01-25": ["НДФЛ", "НДС"],
        "2018-02-15": ["Налог на землю"],
        "2019-04-25": ["НДС"],
        "2019-07-01": ["Транспортный налог"],
        "2020-07-15": ["НДФЛ"],
        "2021-03-15": ["НДФЛ", "Страховые взносы", "Налог на прибыль"],
    }

    def fetch_calendar(self, year):
        url = f"https://isdayoff.ru/api/getdata?year={year}&pre=1"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Ошибка при запросе к API за {year} год: {response.status_code}")
        return response.text

    def get_tax_info_for_date(self, date_str):
        taxes_today = self.TAX_EVENTS.get(date_str, [])
        return {tax: int(tax in taxes_today) for tax in self.TAX_TYPES}

    def create_calendar_dataframe(self):
        rows = []

        for year in range(self.START_YEAR, self.END_YEAR + 1):
            print(f"Обработка {year} года...")
            calendar_data = self.fetch_calendar(year)
            start_date = date(year, 1, 1)

            for i, status in enumerate(calendar_data):
                current_date = start_date + timedelta(days=i)
                current_date_str = str(current_date)

                day_type = "Рабочий" if status == "0" else "Выходной"
                is_working = 1 if status == "0" else 0

                tax_info = self.get_tax_info_for_date(current_date_str)
                row = {
                    "date": current_date_str,
                    "Тип дня": day_type,
                    "Рабочий день (1/0)": is_working,
                    **tax_info
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        self.calendar_df = df
        return df

    def get_calendar(self):
        if self.calendar_df is None:
            return self.create_calendar_dataframe()
        return self.calendar_df
#  Project 2_2023.xlsx'
    def load_and_process_proj_file(self, file_path='../Project 2_2023.xlsx'):
        """
        Метод для загрузки Excel файла, обработки данных и вывода первых строк.
        """
        # Загружаем Excel файл
        self.proj_file = pd.read_excel(file_path)

        # Удаляем столбец с названием 'Unnamed: 0', если он есть
        self.proj_file = self.proj_file.drop('Unnamed: 0', axis=1, errors='ignore')

        # Преобразуем столбец 'date' в формат datetime
        self.proj_file['date'] = pd.to_datetime(self.proj_file['date'])

        # Преобразуем в строковый формат даты
        self.proj_file['date'] = self.proj_file['date'].dt.strftime('%Y-%m-%d')



        return self.proj_file


    def combine_and_export_data(self, proj_file_data, calendar_df, df_daily_digitalization_level, df_daily_age_structure, df_daily_population_density,
                                consumer_expenses_df, df_daily_gdp_per_capita, df_daily_unemployment_rate, df_daily_trade_balance,
                                df_daily_inflation_rate, full_df, df_data, output_file="final.csv"):

        # Объединяем данные по ключу 'date'
        df_combined = pd.merge(proj_file_data, calendar_df, on='date')
        df_combined = pd.merge(df_combined, df_daily_digitalization_level, on='date')
        df_combined = pd.merge(df_combined, df_daily_age_structure, on='date')
        df_combined = pd.merge(df_combined, df_daily_population_density, on='date')
        df_combined = pd.merge(df_combined, consumer_expenses_df, on='date')
        df_combined = pd.merge(df_combined, df_daily_gdp_per_capita, on='date')
        df_combined = pd.merge(df_combined, df_daily_unemployment_rate, on='date')
        df_combined = pd.merge(df_combined, df_daily_trade_balance, on='date')
        df_combined = pd.merge(df_combined, df_daily_inflation_rate, on='date')
        df_combined = pd.merge(df_combined, full_df, on='date')
        df_combined = pd.merge(df_combined, df_data, on='date')

        # Сохраняем объединённый DataFrame в CSV файл
        df_combined.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Data has been merged and saved to {output_file}")