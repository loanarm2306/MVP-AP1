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

La aplicación web se ha desplegado en Streamlit Cloud y se encuentra disponible en:

https://mvp-ap1-4pfz2kgubnhcb94vfec3op.streamlit.app/

La aplicación consume la API publicada en Render:

https://mvp-ap.onrender.com

Para producción, la aplicación utiliza la variable/secreto:

API_URL=https://mvp-ap.onrender.com