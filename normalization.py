import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('../GoldVisionary/altin_verileri.csv')
df.dropna(inplace=True)
df['Tarih'] = pd.to_datetime(df['Tarih'], format='%d.%m.%Y')
df.sort_values(by='Tarih', inplace=True)
df['Fiyat'] = df['Fiyat'].str.replace('.', '').str.replace(',', '.').astype(float)
df.set_index('Tarih', inplace=True)
print(df.info())
print(df.describe())
df.to_csv('cleaned_data.csv')

df = pd.read_csv('../GoldVisionary/cleaned_data.csv')
# RSI hesaplama
def hesapla_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['EMA_14'] = df['Fiyat'].ewm(span=14, adjust=False).mean()  # 14 günlük EMA
df['SMA_50'] = df['Fiyat'].rolling(window=50).mean()  # 50 günlük SMA
df['SMA_200'] = df['Fiyat'].rolling(window=200).mean()  # 200 günlük SMA
df['RSI_14'] = hesapla_rsi(df['Fiyat'], window=14)  # RSI hesaplama
# Bollinger Bands hesaplama
df['SMA_20'] = df['Fiyat'].rolling(window=20).mean()
df['std_dev'] = df['Fiyat'].rolling(window=20).std()
df['Upper_Band'] = df['SMA_20'] + (df['std_dev'] * 2)
df['Lower_Band'] = df['SMA_20'] - (df['std_dev'] * 2)
df.rename(columns={'Tarih': 'date'}, inplace=True)
df.fillna(method='bfill', inplace=True)
print(df)
df.to_csv('indicator_data.csv')

df = pd.read_csv('../GoldVisionary/indicator_data.csv')
scaler = MinMaxScaler()

# İlgili sütunları seçiyoruz.
# (Fiyat ve diğer indikatörler)
columns_to_normalize = ['Fiyat', 'EMA_14', 'SMA_50', 'SMA_200', 'RSI_14', 'SMA_20', 'std_dev', 'Upper_Band', 'Lower_Band']

# Bu sütunları normalize ediyoruz
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Sonuçları kontrol ediyoruz
print(df.head())
df.to_csv('normalized_data.csv',index=False)
print(df.isnull().sum())  # Her sütundaki eksik değer sayısını kontrol edin
