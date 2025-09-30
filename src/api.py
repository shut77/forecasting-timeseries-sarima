from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import numpy as np
import utils
from typing import List, Optional
from pydantic import BaseModel
import model
import pandas as pd
from io import BytesIO


app = FastAPI(title="SARIMAX Forecasting API")

function_forecast = model.load_sarimax_compact(r'C:\Users\Artem\Desktop\vs code project\SARIMA_for_git\models\sarimax_model_clean2.pkl')

# Модель для входных данных
class ForecastRequest(BaseModel):
    steps: int = 24
    exog_data: Optional[List[List[float]]] = None  # экзогенные переменные на будущее

class ForecastResponse(BaseModel):
    forecast: List[float]
    #confidence_intervals: List[dict]
    steps: int

@app.get("/", summary='Начало работы', tags=['просто начало2'])
def home():
    return {"message": "SARIMAX Forecasting API is running", "docs": "/docs"}

@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    """Эндпоинт для прогнозирования"""
    try:
        print(f"🔍 Получен запрос: steps={request.steps}")
        
        # Делаем прогноз
        forecast_result = function_forecast(
            steps=request.steps, 
            exog=np.array(request.exog_data) if request.exog_data else None
        )
        
        print(f"✅ Прогноз создан: {type(forecast_result)}")
        
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        print(f"📊 Результат: {len(forecast_mean)} точек")
        
        # Форматируем ответ
        response = ForecastResponse(
            forecast=forecast_mean.tolist(),
            #confidence_intervals=[
                #{"lower": float(lower), "upper": float(upper)} 
                #for lower, upper in zip(conf_int.iloc[:, 0], conf_int.iloc[:, 1])
            #],
            steps=request.steps
        )
        
        return response
        
    except Exception as e:
        
        print(f" ОШИБКА: {str(e)}")
        import traceback
        print(f" ДЕТАЛИ: {traceback.format_exc()}")
        
        
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "type": type(e).__name__
            }
        )

















@app.post('/model', summary='Для прогноза с экзо')
async def predict_so_big(file: UploadFile = File(...), time_col: str = Form("time"), 
                         value_col: str = Form("value"), train_size: float = Form(0.8), 
                         season_period: int = Form(24), freq: str = Form(None)):
    read = await file.read()
    df = pd.read_csv(BytesIO(read))
    df = utils.setup_time_index(df, time_col)
    series = df[value_col]

    # ПОДГОТАВЛИВАЕМ РЯД И СОЗДАЕМ ЭКЗОГЕННЫЕ ПРИЗНАКИ
    season = season_period
    create_fourier = utils.create_exog_fourier(df, seasonal=season)
    
    if freq is None:
        freq = utils.get_data_frequency(df, time_col=time_col)

    create_calendar_exog = utils.create_exog_calendar(df)
    concat_all_exog = utils.concat_and_create_exog_fourier_for_weekend(create_fourier, create_calendar_exog)
    train, test, exog_train, exog_test  = utils.train_test_split_with_exog(df, concat_all_exog, split=train_size)
    print('Train, test, train_exog, test_exog: ', len(train), len(test), len(exog_train), len(exog_test))
    ex_tr_s, ex_test_s = utils.exog_scaler(exog_train, exog_test)

    #ПОДБИРАЕМ ПАРАМЕТРЫ SARIMAX С ПОМОЩЬЮ AUTOARIMA

    return




















if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)