# MVP-AP1 - Predicción de mora en créditos

Repositorio desarrollado para las Actividades Prácticas I y II de la asignatura **Metodologías de Gestión y Diseño de Proyectos Big Data**, del Máster en Big Data y Data Science.

El proyecto parte de un MVP inicial basado en notebooks y lo transforma en una solución más reproducible, automatizada y desplegable, aplicando prácticas de **CRISP-DM**, **Scrum**, **DataOps** y **MLOps**.

## Autora

**Loana Rodrigues Morais**

## Enlaces del proyecto

- Repositorio GitHub: <https://github.com/loanarm2306/MVP-AP1>
- GitHub Project: <https://github.com/users/loanarm2306/projects/3/views/1>
- API FastAPI en Render: <https://mvp-ap.onrender.com>
- Aplicación web en Streamlit Cloud: <https://mvp-ap1-4pfz2kgubnhcb94vfec3op.streamlit.app/>

## Objetivo del proyecto

El objetivo del proyecto es predecir la posibilidad de que un crédito otorgado por una entidad financiera pueda entrar en situación de mora o falta de pago.

Para ello, se trabaja con dos fuentes de datos:

- Datos de créditos solicitados por los clientes.
- Datos de otros productos financieros del cliente, principalmente tarjetas de crédito.

A partir de estos datos se construye un dataset preparado para modelado, se entrena un modelo de clasificación y se despliega una versión funcional del MVP mediante una API y una aplicación web.

## Tecnologías utilizadas

- Python
- pandas
- scikit-learn
- PyTest
- DVC
- MLflow
- FastAPI
- Streamlit
- Render
- Streamlit Cloud
- GitHub Projects

## Estructura del repositorio

```text
MVP-AP1/
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── despliegue.md
├── models/
│   └── modelo_mora.pkl
├── notebooks/
│   ├── CD01_Visualizacion.ipynb
│   ├── CD02_Procesamiento.ipynb
│   └── MD01_Experimentacion.ipynb
├── reports/
│   ├── figures/
│   ├── metricas_modelo.json
│   ├── preparacion_datos_resumen.txt
│   └── resultados_experimentacion.csv
├── src/
│   ├── generar_visualizaciones.py
│   ├── preparar_datos.py
│   └── entrenar_modelo.py
├── tests/
│   ├── test_calidad_datos.py
│   └── test_modelo.py
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md
