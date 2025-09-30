import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss
import os
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX


#ФУНКЦИИ ДЛЯ API 

def get_data_frequency(df, time_col='timestamp'):
    df[time_col] = pd.to_datetime(df[time_col])
    time_diff = df[time_col].diff().mode().iloc[0] 
    
    
    if time_diff == pd.Timedelta(hours=1):
        return 'H', 1
    elif time_diff == pd.Timedelta(days=1):
        return 'D', 1  
    elif time_diff == pd.Timedelta(hours=24):
        return 'H', 24
    elif time_diff == pd.Timedelta(minutes=30):
        return 'min', 30
    else:
        return f"{time_diff}", time_diff.total_seconds()/3600 
    

def setup_time_index(df, time_col="time"):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


#ФУНКЦИИ ДЛЯ EDA



def visual(df, time_col='time', val_col ='total load actual',  start_points=None, end_points=None, title='Потребление - время', label = 'Потребление', xlabel='Дата', ylabel='МВт' ):
    if time_col not in df.columns:
        time_data = df.index
    else:
        time_data = df[time_col]

    if start_points is None or end_points is None:
        value_data = df[val_col]
    else:
        time_data = time_data[start_points:end_points]
        value_data = df[val_col][start_points:end_points]

    plt.figure(figsize=(18,5))
    plt.plot(time_data, value_data, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def corralelogramm(df, lags_acf = 30, lags_pacf = 30):
    plot_acf(df.dropna(), lags = lags_acf)
    plt.title('АКФ')
    plt.show()
    plot_pacf(df.dropna(), lags = lags_pacf)
    plt.title("ЧАКФ")
    plt.show()

def diff(df, lag = 1):
    df_diff = df.diff(lag).dropna()
    return df_diff

def create_exog_fourier(df, D=3, W=4, ME=4, seasonal=False, order = 0):
    df.index = pd.to_datetime(df.index).tz_localize(None)
    fourier_daily = CalendarFourier(freq='D', order=D)
    fourier_week  = CalendarFourier(freq='W', order=W)
    fourier_month  = CalendarFourier(freq='ME', order=ME)
    dp = DeterministicProcess(
        index=df.index,
        constant=True,
        order=order,
        seasonal=seasonal,                
        additional_terms=[fourier_daily, fourier_week, fourier_month],
        drop=True,
    )
    df_fourier = dp.in_sample()
    return df_fourier

def create_exog_calendar(df, freq='h'):
    df_features = pd.DataFrame(index=df.index).asfreq(freq) 
    df_features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df_features['month'] = df.index.month
    df_features['is_night'] = ((df.index.hour >= 4) & (df.index.hour <= 7)).astype(int)
    df_features['is_evening'] = ((df.index.hour >= 20) | (df.index.hour == 0)).astype(int)
    return df_features

def concat_and_create_exog_fourier_for_weekend(df_fourier, df_features, freq='h'):
    exog = pd.concat([df_fourier, df_features], axis=1)
    exog = exog.dropna()        
    exog['hour'] = exog.index.hour
    hour_dummies = pd.get_dummies(exog['hour'], prefix=freq, drop_first=True)
    fourier_week_cols = [c for c in exog.columns if 'freq=W' in str(c) or 'W-SUN' in str(c)] 
    for c in fourier_week_cols:
        exog[c + '_wkend'] = exog[c] * exog['is_weekend']
    return exog

def train_test_split_with_exog(df, exog, split=0.8, freq='h'):
    df = df.loc[exog.index]
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    exog_train, exog_test = exog.iloc[:train_size], exog.iloc[train_size:]
    train = train.asfreq(freq)
    test = test.asfreq(freq)
    exog_train = exog_train.asfreq(freq)
    exog_test = exog_test.asfreq(freq)
    return train, test, exog_train, exog_test 

def exog_scaler(exog_train, exog_test):
    sc = StandardScaler()
    exog_train_s = pd.DataFrame(sc.fit_transform(exog_train), index=exog_train.index, columns=exog_train.columns)
    exog_test_s  = pd.DataFrame(sc.transform(exog_test), index=exog_test.index, columns=exog_test.columns)
    return exog_train_s, exog_test_s

def static_tests(df, regression='c'):
    result_ADF = adfuller(df.interpolate(), regression=regression)
    print("ADF Statistic:", result_ADF[0])
    print("p-value:", result_ADF[1])
    for key, value in result_ADF[4].items():
        print("Critical Value (%s): %.3f" % (key, value))
    
    if result_ADF[1] <= 0.05:  
        print("Результат: p-value <= 0.05 -> Отвергаем H0. Ряд СТАЦИОНАРЕН.")
    else:  
        print("Результат: p-value > 0.05 -> Не можем отвергнуть H0. Ряд НЕ стационарен.")


    kpss_stat, p_value, n_lags, critical_values = kpss(df.interpolate(), regression=regression)
    print(f'\nKPSS Statistic: {kpss_stat}')
    print(f'p-value: {p_value}')
    print(f'Number of Lags: {n_lags}')
    for key, value in critical_values.items():
        print("Critical Value (%s): %.3f" % (key, value))

    if p_value < 0.05:
        print("Результат: p-value < 0.05 -> Отвергаем H0. Ряд НЕ стационарен.")
    else:
        print("Результат: p-value >= 0.05 -> Не можем отвергнуть H0. Ряд стационарен.")


