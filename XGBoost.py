import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, DMatrix, train
import matplotlib
matplotlib.use('Agg')

data = pd.read_csv('indicator_data.csv')
df = data.copy()
df['preds'] = df['Fiyat'].shift(-3)  # num=3
numeric_features = ['EMA_14', 'SMA_50', 'SMA_200', 'RSI_14',
                    'SMA_20', 'std_dev', 'Upper_Band', 'Lower_Band']

x = df[numeric_features].values
y = df['preds'].values[:-3]
x = x[:-3]
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=7
)
start_train_time = time.time()  # Başlama zamanı
# Eğitim işlemi
eval_set = [(x_train, y_train), (x_test, y_test)]
xgb_model.fit(
    x_train, y_train,
    eval_set=eval_set,
    verbose=False
)
end_train_time = time.time()  # Bitiş zamanı
training_time = end_train_time - start_train_time
print(f"Eğitim Süresi: {training_time:.2f} saniye")
# Modelin tahmin yapması
y_pred = xgb_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r_squared = r2_score(y_test, y_pred)

# Performans metrikleri
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R-Squared: {r_squared:.2f}')

start_inference_time = time.time()  # Başlama zamanı
# Gelecek günlerin tahmini
x_forecast = x[-3:]
forecasted_pred = xgb_model.predict(x_forecast)
# for day, price in enumerate(forecasted_pred, start=1):
    # print(f'{day}. Gün için tahmin edilen fiyat: {price:.2f}')
end_inference_time = time.time()  # Bitiş zamanı
inference_time = end_inference_time - start_inference_time
print(f"Çıkarım Süresi: {inference_time:.5f} saniye")
# Loss grafiği (Eğitim ve Doğrulama)
evals_result = xgb_model.evals_result()
# Epoch ve loss değerlerini çekelim
epochs = range(1, len(evals_result['validation_0']['rmse']) + 1)
training_loss = evals_result['validation_0']['rmse']  # Eğitim seti loss'u (örnek: RMSE)
validation_loss = evals_result['validation_1']['rmse']  # Doğrulama seti loss'u
# Grafiği çizelim
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_loss, label='Eğitim Loss (RMSE)')
plt.plot(epochs, validation_loss, label='Doğrulama Loss (RMSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (RMSE)')
plt.title('Epoch ve Loss Grafiği')
plt.legend()
plt.grid()
plt.savefig('XGBoost_plot.png')  # Grafiği kaydeder
plt.close()
