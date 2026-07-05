# Despliegue del MVP

## API FastAPI

La API del proyecto se ha desarrollado con FastAPI y se despliega como servicio web en Render.

URL de producción:

https://mvp-ap.onrender.com

Endpoints principales:

- `/`: estado general de la API.
- `/health`: comprobación de disponibilidad.
- `/features`: listado de variables requeridas por el modelo.
- `/predict`: endpoint para generar predicciones.
- `/docs`: documentación interactiva generada automáticamente por FastAPI.

Comando local de ejecución:

uvicorn app.api:app --reload

Comando de producción utilizado en Render:

uvicorn app.api:app --host 0.0.0.0 --port $PORT

## Aplicación Streamlit

La aplicación web se ha desarrollado con Streamlit y permite a los usuarios introducir valores de las variables del modelo para obtener una predicción de riesgo de mora.

Comando local de ejecución:

streamlit run app/streamlit_app.py

Para el despliegue en Streamlit Cloud, la aplicación debe configurar la siguiente variable o secreto:

API_URL=https://mvp-ap.onrender.com

De esta forma, la interfaz web consulta la API publicada en Render y utiliza el modelo desplegado para generar predicciones.