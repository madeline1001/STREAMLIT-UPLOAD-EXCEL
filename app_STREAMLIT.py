# Importamos las bibliotecas necesarias
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.regression import predict_model
import tempfile

# Cargar el modelo preentrenado desde el archivo pickle
model_path = "best_model.pkl"
with open(model_path, 'rb') as model_file:
    modelo = pickle.load(model_file)

# Título de la API
st.title("API de Predicción Precio")

# Botón para subir archivo Excel
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx", "csv"])

prueba = pd.read_csv("prueba_APP.csv",header = 0,sep=";",decimal=",")

# Botón para predecir
if st.button("Predecir"):
    if uploaded_file is not None:
        try:
            # Cargar el archivo subido
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            if uploaded_file.name.endswith(".csv"):
                prueba = pd.read_csv(tmp_path,header = 0,sep=";",decimal=",")
            else:
                prueba = pd.read_excel(tmp_path,header = 0,sep=";",decimal=",")

             # Realizar predicción
            predictions = predict_model(modelo, data=prueba)
            predictions["price"] = predictions["prediction_label"]

            # Preparar archivo para descargar
            kaggle = pd.DataFrame({'Email': prueba["Email"], 'price': predictions["price"]})

            # Mostrar predicciones en pantalla
            st.write("Predicciones generadas correctamente!")
            st.write(kaggle)

            # Botón para descargar el archivo de predicciones
            st.download_button(label="Descargar archivo de predicciones",
                               data=kaggle.to_csv(index=False),
                               file_name="kaggle_predictions.csv",
                               mime="text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Por favor, cargue un archivo válido.")

# Botón para reiniciar la página
if st.button("Reiniciar"):
    st.experimental_rerun()

            