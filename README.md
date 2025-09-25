# Linear Forecasting with SARIMAX

## 📌 Описание
Этот проект посвящён анализу и прогнозированию временных рядов энергопотребления.  
На текущем этапе выполнен исследовательский анализ данных:
- визуализация ряда,
- проверка стационарности с помощью тестов ADF, KPSS,
- автокорреляционные функции (ACF/PACF),
- дифференцирование,
- формирование экзогенных переменных,
- разделение на train/test,
- стандартизация экзогенных переменных.
- сохранение готовых данных после EDA.

## 🗂️ Структура проекта
- `notebooks/analysis.ipynb` — исследовательский анализ, визуализация.
- `src/utils.py` — функции анализа.
- `data/` — данные 
- `reports/` — графики и отчёты.
- `requirements.txt` — зависимости.

## 🚀 Запуск

1. Склонировать проект:
```bash
git clone https://github.com/username/linear-forecasting-sarima.git
cd linear-forecasting-sarima
```

2. Установить зависимости:
```bash
pip install -r requirements.txt
```

3. Запустить ноутбук:
```bash
jupyter notebook notebooks/analysis.ipynb
```