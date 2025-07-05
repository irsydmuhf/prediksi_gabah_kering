import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@st.cache_data
def load_data():
    data = pd.read_csv('hasilPreProcessing.csv')
    data['sin_bulan'] = np.sin(2 * np.pi * data['Bulan'] / 12)
    data['cos_bulan'] = np.cos(2 * np.pi * data['Bulan'] / 12)
    return data

data = load_data()

def main():
    st.title("Dashboard Prediksi Harga Gabah Kering Panen")

    st.sidebar.header("Pengaturan")
    prediction_date = st.sidebar.date_input("Pilih tanggal prediksi")

    st.write("### Dataset")
    st.dataframe(data[['Bulan', 'Harga']], use_container_width=True)

    X = data[['Bulan', 'sin_bulan', 'cos_bulan']]
    y = data['Harga']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    tanggal_awal = datetime(2011, 1, 1)
    selisih_bulan = (prediction_date.year - tanggal_awal.year) * 12 + (prediction_date.month - tanggal_awal.month) + 1
    sin_bulan = np.sin(2 * np.pi * selisih_bulan / 12)
    cos_bulan = np.cos(2 * np.pi * selisih_bulan / 12)
    bulan_baru = pd.DataFrame({'Bulan': [selisih_bulan], 'sin_bulan': [sin_bulan], 'cos_bulan': [cos_bulan]})

    lr_prediction = lr_model.predict(bulan_baru)[0]
    rf_prediction = rf_model.predict(bulan_baru)[0]

    st.write("### Hasil Prediksi")
    st.write(f"**Tanggal Prediksi:** {prediction_date}")
    st.write(f"**Prediksi Harga (Linear Regression):** Rp {lr_prediction:,.2f}")
    st.write(f"**Prediksi Harga (Random Forest):** Rp {rf_prediction:,.2f}")

    y_pred_lr = lr_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    st.write("### Evaluasi Model")
    st.write(f"**Akurasi Linear Regression:** {r2_lr:.2f}")
    st.write(f"**MSE Linear Regression:** {mse_lr:.2f}")
    st.write(f"**MAE Linear Regression:** {mae_lr:.2f}")
    st.write(f"**Akurasi Random Forest:** {r2_rf:.2f}")
    st.write(f"**MSE Random Forest:** {mse_rf:.2f}")
    st.write(f"**MAE Random Forest:** {mae_rf:.2f}")

if __name__ == "__main__":
    main()
