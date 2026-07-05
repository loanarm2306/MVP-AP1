import os

import requests
import streamlit as st

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://127.0.0.1:8000"))

st.set_page_config(
    page_title="Predicción de mora",
    page_icon="📊",
    layout="centered",
)

st.title("Predicción de mora en créditos")
st.write(
    "Aplicación web para consultar el modelo de predicción de riesgo de mora "
    "a partir de los datos de un cliente y su crédito."
)

st.sidebar.header("Configuración")
api_url = st.sidebar.text_input("URL de la API", value=API_URL)

st.subheader("Estado de la API")

try:
    health = requests.get(f"{api_url}/health", timeout=5)
    if health.status_code == 200:
        st.success("API conectada correctamente.")
    else:
        st.warning("La API respondió, pero no con estado correcto.")
except Exception as e:
    st.error("No se pudo conectar con la API.")
    st.stop()

try:
    response = requests.get(f"{api_url}/features", timeout=10)
    response.raise_for_status()
    info_modelo = response.json()
    columnas = info_modelo["feature_columns"]

    st.subheader("Información del modelo")
    st.write(f"Modelo seleccionado: **{info_modelo['selected_model']}**")
    st.write(f"Variable objetivo: **{info_modelo['target']}**")

except Exception as e:
    st.error(f"No se pudieron cargar las columnas del modelo: {e}")
    st.stop()

st.subheader("Datos para la predicción")

st.write(
    "Introduce los valores de las variables disponibles. "
    "Los campos que no se modifiquen se enviarán con valor 0."
)

features = {}

with st.form("formulario_prediccion"):
    for columna in columnas:
        features[columna] = st.number_input(
            columna,
            value=0.0,
            step=1.0,
            format="%.4f",
        )

    submitted = st.form_submit_button("Generar predicción")

if submitted:
    try:
        payload = {"features": features}
        pred_response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=10,
        )
        pred_response.raise_for_status()
        resultado = pred_response.json()

        st.subheader("Resultado de la predicción")

        if resultado["prediccion"] == 1:
            st.error(resultado["resultado"])
        else:
            st.success(resultado["resultado"])

        st.write(f"Predicción numérica: **{resultado['prediccion']}**")

        if resultado["probabilidad_mora"] is not None:
            st.write(
                f"Probabilidad estimada de mora: "
                f"**{resultado['probabilidad_mora']:.2%}**"
            )

    except Exception as e:
        st.error(f"No se pudo generar la predicción: {e}")