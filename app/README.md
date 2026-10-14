# Árbol de decisión · Accidentes aéreos

App web en Streamlit (modelo: DecisionTreeClassifier).

## Ejecutar en local
    pip install -r requirements.txt
    streamlit run app.py

## Publicar con un enlace (Streamlit Community Cloud)
1. Sube esta carpeta a un repositorio de GitHub.
2. En share.streamlit.io elige el repo y `app.py` como archivo principal → Deploy.

## Archivos
- `app.py`: interfaz (guía, datos, evaluación, árbol, simulador, conclusiones)
- `modelo.py`: limpieza, entrenamiento, GridSearchCV y métricas (código comentado)
- `data/plane_crashes_data.csv`: dataset
