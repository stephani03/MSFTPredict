#Library
import pandas as pd
import pickle
import streamlit as st
from sklearn.metrics import mean_squared_error


#Dataset
def load_data(ticker, start_date, end_date):
    data = pd.read_csv(r'C:\Users\Stephani G\Documents\hani\college hani\stupen\tasks\Final Project\MSFT.csv')
    return data

#Mempersiapkan data prediksi
def prepare_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(pd.Timestamp.timestamp)
    x = data[['Open','High','Low','Volume']].values
    y = data['Close'].values
    return x,y

#Membuat model dari file Pickle
def load_model():
    model = pickle.load(open('Final_Project.sav','rb'))
    return model

#Streamlit
def main():
    st.title("Prediksi Harga Saham MSFT")
    
    ticker = st.text_input("Masukkan Ticker Saham", "MSFT")
    start_date = st.date_input("Pilih Tanggal Mulai", pd.to_datetime("2023-08-01"))
    end_date = st.date_input("Pilih Tanggal Akhir", pd.to_datetime("2023-09-01"))
    
    if st.button('Prediksi'):
        predictions = model.predict([ticker,start_date,end_date])

        if ticker and start_date and end_date:
            data = load_data(ticker, start_date, end_date)
        
        if not data.empty:
            st.subheader("Data Historis Saham")
            st.write(data.tail())
            
            x, y = prepare_data(data)
            model = load_model()
            predictions = model.predict(x)
            
            st.subheader("Prediksi vs Aktual")
            results = pd.DataFrame({"Tanggal": pd.to_datetime(data['Date'], unit='s'), "Aktual": y, "Prediksi": predictions})
            st.write(results.set_index("Tanggal"))
            
            mse = mean_squared_error(y, predictions)
            st.write(f"Mean Squared Error: {mse}")
        else:
            st.write("Data tidak tersedia untuk ticker dan rentang tanggal yang dipilih.")

if __name__ == "__main__":
    main()
