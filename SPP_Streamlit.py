#7. Membuat Model untuk Aplikasi

import pandas as pd
import pickle
import streamlit as st
from datetime import datetime

# Dataset
MSFT_Dataset = 'MSFT.csv'
def load_data(MSFT_Dataset):
    data = pd.read_csv(r'data/MSFT.csv')
    return data

# Mempersiapkan data untuk prediksi
def prepare_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Timestamp'] = data['Date'].apply(lambda x: datetime.timestamp(x))
    x = data[['Open', 'High', 'Low','Volume']]
    y = data['Close']
    return x, y

# Memuat model dari file Pickle
def load_model_and_scaler():
    with open('stock_model.pkl','rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl','rb') as file:
        scaler = pickle.load(file)
    return model, scaler

# Streamlit
def main():
    st.title("Prediksi Harga Saham MSFT")
    
    # Menentukan path file CSV
    MSFT_Dataset = 'data/MSFT.csv'

    # Memuat data dari file CSV
    data = load_data(MSFT_Dataset)

    if not data.empty:
        st.subheader("Data Historis Saham")
        st.write(data.tail())
        
        x,y = prepare_data(data)
        model,scaler = load_model_and_scaler()
        
        # Menentukan batasan tanggal
        min_date = datetime.strptime('1980-01-01', '%Y-%m-%d')
        max_date = datetime.strptime('2050-12-31', '%Y-%m-%d')

        # Input
        prediction_date = st.date_input("Masukkan Tanggal untuk Prediksi", datetime.today(), min_value=min_date, max_value=max_date)

        open_value = st.number_input('Masukkan nilai Open', value=float(data['Open'].iloc[-1]))
        high_value = st.number_input('Masukkan nilai High', value=float(data['High'].iloc[-1]))
        low_value = st.number_input('Masukkan nilai Low', value=float(data['Low'].iloc[-1]))
        volume_value = st.number_input('Masukkan nilai Volume', value=float(data['Volume'].iloc[-1]))
        
        input_data = [[open_value, high_value, low_value, volume_value]]
        scaled_input_data = scaler.transform(input_data)
        
        # Prediksi menggunakan model
        prediction = model.predict(scaled_input_data)

        st.subheader(f"Prediksi Harga Saham MSFT untuk {prediction_date}")
        st.write(f"Prediksi Harga Penutupan: ${prediction[0]:.2f}")
    else:
        st.write("Data tidak tersedia.")

if __name__ == "__main__":
    main()