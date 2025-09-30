import pandas as pd 
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np


#ЗАГРУЗКА/ВЫГРУЗКА ДАННЫХ
def save_data(train, test, ex_tr, ex_test, folder='../data'):
    os.makedirs(folder, exist_ok=True)
    
    train.to_pickle(f'{folder}/train.pkl')
    test.to_pickle(f'{folder}/test.pkl')
    ex_tr.to_pickle(f'{folder}/ex_tr.pkl')
    ex_test.to_pickle(f'{folder}/ex_test.pkl')
    
    print(f"Данные сохранены в папку {folder}")

def load_data(folder='../data'):
    train = pd.read_pickle(f'{folder}/train.pkl')
    test = pd.read_pickle(f'{folder}/test.pkl')
    ex_tr = pd.read_pickle(f'{folder}/ex_tr.pkl')
    ex_test = pd.read_pickle(f'{folder}/ex_test.pkl')
    
    print("Данные загружены!")
    return train, test, ex_tr, ex_test

#ФУНКЦИИ ДЛЯ МОДЕЛИ, МЕТРИКИ
def monitoring_callback(params):
    print(f"Iteration: {monitoring_callback.iteration}")
    monitoring_callback.iteration += 1

def metrics(test, forecast, exp=False):
    if isinstance(test, pd.DataFrame):
        test = test.iloc[:, 0]  
    if isinstance(forecast, pd.DataFrame):
        forecast = forecast.iloc[:, 0]  
    
    valid_idx = test.notna() & forecast.notna()
    
    test = test[valid_idx]
    forecast = forecast[valid_idx]

    cumulative_mae = []
    for i in range(2, len(test) + 1):
        if exp is False:
            mae = mean_absolute_error(test[:i], forecast[:i])
            cumulative_mae.append(mae)
        else:
            mae = mean_absolute_error(np.exp(test[:i]), np.exp(forecast[:i]))
            cumulative_mae.append(mae)
    print(f'Метрика MAE: {mae}')
    plt.figure(figsize=(12, 6))
    plt.axvline(x=24*90, color='red', linestyle='--', linewidth=2, label='90 дней')
    plt.plot(range(2, len(test) + 1), cumulative_mae, marker='o')
    plt.xlabel('Количество объектов в тестовой выборке')
    plt.ylabel('MAE')
    plt.title('Зависимость MAE от размера тестовой выборки')
    plt.grid(True)
    plt.show()
    
    cumulative_R2_score = []
    for i in range(2, len(test) + 1):
        if exp is False:
            R2_score = r2_score(test[:i], forecast[:i])
            cumulative_R2_score.append(R2_score)
        else:
            R2_score = r2_score(np.exp(test[:i]), np.exp(forecast[:i]))
            cumulative_R2_score.append(R2_score)
    print(f'Метрика R2_score: {R2_score}')
    plt.figure(figsize=(12, 6))
    plt.axvline(x=24*90, color='red', linestyle='--', linewidth=2, label='90 дней')
    plt.plot(range(2, len(test) + 1), cumulative_R2_score, marker='o')
    plt.xlabel('Количество объектов в тестовой выборке')
    plt.ylabel('R2_score')
    plt.title('Зависимость R2_score от размера тестовой выборки')
    plt.grid(True)
    plt.show()

    cumulative_MAPE = []
    for i in range(2, len(test) + 1):
        if exp is False:
            mape = mean_absolute_percentage_error(test[:i], forecast[:i])
            cumulative_MAPE.append(mape)
        else:
            mape = mean_absolute_percentage_error(np.exp(test[:i]), np.exp(forecast[:i]))
            cumulative_MAPE.append(mape)
    print(f'Метрика MAPE: {mape}')
    plt.figure(figsize=(12, 6))
    plt.axvline(x=24*90, color='red', linestyle='--', linewidth=2, label='90 дней')
    plt.plot(range(2, len(test) + 1), cumulative_MAPE, marker='o')
    plt.xlabel('Количество объектов в тестовой выборке')
    plt.ylabel('MAPE')
    plt.title('Зависимость MAPE от размера тестовой выборки')
    plt.grid(True)
    plt.show()
    
    return mae, R2_score, mape

#СОХРАНЕНИЕ МОДЕЛИ
def save_sarimax_compact(results, filepath, train, exog=None):
    """Сохраняем параметры и часть данных для корректного прогноза"""
    p, d, q = results.model.order
    P, D, Q, s = results.model.seasonal_order
    
    # сколько точек нужно для инициализации
    required_lags = max(
        p + s * P,      # AR + seasonal AR компоненты
        q + s * Q,      # MA + seasonal MA компоненты  
        d + s * D,      # дифференцирования
        2 * s           # минимум 2 сезона для стабильности
    )
    
    # Берем последние точки
    if hasattr(train, 'iloc'):  # если DataFrame
        train_tail = train.iloc[-required_lags:].values.flatten().tolist()
    elif hasattr(train, 'values'):  # если Series
        train_tail = train.values[-required_lags:].tolist()
    else:  # если numpy array
        train_tail = train[-required_lags:].tolist()


    compact_data = {
        "params": results.params,
        "param_names": results.param_names,
        "order": results.model.order,
        "seasonal_order": results.model.seasonal_order,
        "enforce_stationarity": results.model.enforce_stationarity,
        "enforce_invertibility": results.model.enforce_invertibility,
        "nobs": results.nobs,
        "train_tail": train_tail,  
        "exog_shape": exog.shape if exog is not None else None,
    }
    with open(filepath, "wb") as f:
        pickle.dump(compact_data, f)
    print(f"Сохранено: {filepath}")



# ЗАГРУЗКА МОДЕЛИ
def load_sarimax_compact(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    def make_forecast(steps, exog=None):
        # "хвост" ряда для корректных лагов
        y_init = np.array(data["train_tail"] + [0]*steps)

        temp_model = SARIMAX(
            y_init,
            exog=exog,
            order=data["order"],
            seasonal_order=data["seasonal_order"],
            enforce_stationarity=data["enforce_stationarity"],
            enforce_invertibility=data["enforce_invertibility"],
        )
        temp_results = temp_model.smooth(data["params"])
        forecast = temp_results.get_forecast(steps=steps, exog=exog)
        #forecast_final, conf_final = forecast.predicted_mean, forecast.conf_int()
        return forecast

    return make_forecast
