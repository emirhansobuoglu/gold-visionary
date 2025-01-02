import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')

# Veriyi yükleyelim ve fiyat sütununu alalım
data = pd.read_csv('cleaned_data.csv')
df = data.copy()
df = df.dropna()

# Tarih sütununu datetime formatına dönüştür
df['Tarih'] = pd.to_datetime(df['Tarih'])
df.set_index('Tarih', inplace=True)  # Tarih indeks olarak ayarlayalım

# Zaman serisi sıklığı belirtelim (örneğin: günlük)
df = df.asfreq('D')

price_data = df['Fiyat'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_price_data = scaler.fit_transform(price_data)

train_size = int(len(scaled_price_data) * 0.8)
train_data = scaled_price_data[:train_size]
test_data = scaled_price_data[train_size:]
test_data = test_data[~np.isnan(test_data).any(axis=1)]

# ARIMA modelini tanımlayalım (p,d,q parametrelerini deneysel olarak ayarlayın)
start_train_time = time.time()  # Başlama zamanı
model = ARIMA(train_data, order=(5, 1, 2))
model_fit = model.fit()
end_train_time = time.time()  # Bitiş zamanı
training_time = end_train_time - start_train_time
print(f"Eğitim Süresi: {training_time:.2f} saniye")

epochs = list(range(1, len(test_data) + 1))  # Epoch sayısı kadar liste oluştur
loss_values = []

# Her adımda tahmin yapıp loss hesapla
for epoch in epochs:
    y_pred_scaled = model_fit.forecast(steps=epoch)  # Tahmin
    loss = np.mean((y_pred_scaled - test_data[:epoch].flatten()) ** 2)  # Kayıp hesapla (MSE)
    loss_values.append(loss)

# Modeli tahmin edelim
start_inference_time = time.time()  # Başlama zamanı
y_pred_scaled = model_fit.forecast(steps=len(test_data))
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
end_inference_time = time.time()  # Bitiş zamanı
# print(f'1. Gün için tahmin edilen fiyat: {y_pred[0][0]:.2f}')
inference_time = end_inference_time - start_inference_time
print(f"Çıkarım Süresi: {inference_time:.5f} saniye")
# Performans metriklerini hesaplayalım
mse = mean_squared_error(test_data, y_pred_scaled)
mae = mean_absolute_error(test_data, y_pred_scaled)
rmse = np.sqrt(mse)
r_squared = r2_score(test_data, y_pred_scaled)

print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R-Squared: {r_squared:.2f}')

# Epoch ve Loss grafiğini çizelim
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, label='Loss', color='red')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('Arima_plot.png')
plt.close()

