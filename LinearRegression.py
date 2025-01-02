from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
import time


def model_engine(model, num):
    # Verilerin hazırlanması
    df = data.copy()
    df['preds'] = df['Fiyat'].shift(-num)
    numeric_features = ['EMA_14', 'SMA_50', 'SMA_200', 'RSI_14',
                        'SMA_20', 'std_dev', 'Upper_Band', 'Lower_Band']
    x = df[numeric_features].values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df['preds'].values
    y = y[:-num]

    # Eğitim ve test verilerinin ayrılması
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

    start_train_time = time.time()  # Başlama zamanı
    # Modelin eğitilmesi
    model.fit(x_train, y_train)
    end_train_time = time.time()  # Bitiş zamanı
    training_time = end_train_time - start_train_time
    print(f"Eğitim Süresi: {training_time:.2f} saniye")

    # Tahminler ve metrikler
    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    r_squared = r2_score(y_test, preds)

    print(f'MSE: {mse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'R-Squared: {r_squared:.2f}')

    # Gelecek tahminler
    start_inference_time = time.time()  # Başlama zamanı
    forecasted_pred = model.predict(x_forecast)
    day = 1
    for price in forecasted_pred:
        # print(f'{day}. Gün için tahmin edilen fiyat: {price:.2f}')
        day += 1
    end_inference_time = time.time()  # Bitiş zamanı
    inference_time = end_inference_time - start_inference_time
    print(f"Çıkarım Süresi: {inference_time:.5f} saniye")


# Veri ve model tanımlama
data = pd.read_csv('indicator_data.csv')
scaler = MinMaxScaler()
num = 3

engine = LinearRegression()
model_engine(engine, num)
