import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time


# Veriyi yükleme ve ön işleme
data = pd.read_csv('indicator_data.csv')
df = data.copy()

# Hedef değeri kaydırma
df['preds'] = df['Fiyat'].shift(-3)  # Tahmin hedefi (3 gün ileri)
numeric_features = ['EMA_14', 'SMA_50', 'SMA_200', 'RSI_14', 'SMA_20', 'std_dev', 'Upper_Band', 'Lower_Band']

# Özellikler ve hedef değişken
x = df[numeric_features].values[:-3]
y = df['preds'].values[:-3]

# Verileri normalizasyon (StandardScaler kullanımı)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Eğitim ve test setlerini ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Zaman serisi verisi oluşturma
sequence_length = 20  # Daha uzun geçmiş kullanımı
n_features = x_train.shape[1]

def create_sequences(data, labels, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        label = labels[i+sequence_length]
        sequences.append((seq, label))
    return zip(*sequences)

X_train_seq, y_train_seq = create_sequences(x_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(x_test, y_test, sequence_length)

# TFT Modelini Tanımlama
def build_tft_model(sequence_length, n_features):
    inputs = Input(shape=(sequence_length, n_features))
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)
    output = Dense(1)(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='huber')  # Huber Loss kullanımı
    return model

# TFT Modelini oluştur
tft_model = build_tft_model(sequence_length, n_features)

# Erken durdurma tanımlama
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
start_train_time = time.time()
# Modeli eğitme
history = tft_model.fit(
    np.array(X_train_seq),
    np.array(y_train_seq),
    epochs=77,  # Daha fazla epoch
    batch_size=16,  # Daha küçük batch size
    validation_data=(np.array(X_test_seq), np.array(y_test_seq)),
    callbacks=[early_stopping]
)
end_train_time = time.time()  # Bitiş zamanı
training_time = end_train_time - start_train_time
print(f"Eğitim Süresi: {training_time:.2f} saniye")
# Performans metriklerini hesaplama
y_pred = tft_model.predict(np.array(X_test_seq))
mse = mean_squared_error(y_test_seq, y_pred)
mae = mean_absolute_error(y_test_seq, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test_seq - y_pred) / y_test_seq)) * 100
r_squared = r2_score(y_test_seq, y_pred)
# 1. gün için tahmin değeri
start_inference_time = time.time()  # Başlama zamanı
first_day_input = np.array(X_test_seq[0])  # İlk sıralı veri
first_day_input = first_day_input.reshape(1, sequence_length, n_features)  # Model girişine uygun boyut
# Tahmin
first_day_prediction = tft_model.predict(first_day_input)
print(f"1. Gün için Tahmin Değeri: {first_day_prediction[0][0]:.2f}")
end_inference_time = time.time()  # Bitiş zamanı
inference_time = end_inference_time - start_inference_time
print(f"Çıkarım Süresi: {inference_time:.5f} saniye")
# Performans metrikleri
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R-Squared: {r_squared:.2f}')

# Loss grafiği (Eğitim ve Doğrulama)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Doğrulama Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch ve Loss Grafiği')
plt.legend()
plt.grid()
plt.savefig('tft_loss_plot_updated.png')
plt.close()
