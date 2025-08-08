import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# === Load pipeline & model ===
preprocessing = joblib.load("pipeline_preprocessing.pkl")
knn_model = joblib.load("knn_model.pkl")

try:
    label_encoder = joblib.load("label_encoder.pkl")
except:
    label_encoder = None

# === UI ===
st.set_page_config(page_title="Prediksi KNN", page_icon="🔮", layout="centered")
st.title("🔮 Prediksi Profitabilitas Menu Restoran")
st.markdown("Masukkan detail menu untuk memprediksi **profitabilitas** berdasarkan model KNN yang telah dilatih.")

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        restaurant_id = st.text_input("RestaurantID")
        menu_category = st.text_input("MenuCategory")
        price = st.number_input("Price", min_value=0.0, format="%.2f")

    with col2:
        ingredients = st.text_area("Ingredients (pisahkan dengan spasi)")
        menu_item = st.text_input("Menu Item")

    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Pastikan nama kolom sesuai pipeline training
    data_input = pd.DataFrame([{
        "RestaurantID": restaurant_id,
        "MenuCategory": menu_category,
        "Ingredients": ingredients,
        "MenuItem": menu_item,
        "Price": price
    }])

    try:
        X_transformed = preprocessing.transform(data_input)
        y_pred = knn_model.predict(X_transformed)

        if label_encoder:
            hasil = label_encoder.inverse_transform(y_pred)[0]
        else:
            hasil = y_pred[0]

        st.success(f"💡 Prediksi: **{hasil}**")

    except Exception as e:

        st.error(f"Terjadi kesalahan: {e}")
