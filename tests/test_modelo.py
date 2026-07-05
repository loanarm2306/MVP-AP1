import json
from pathlib import Path

MODEL_PATH = Path("models/modelo_mora.pkl")
METRICS_PATH = Path("reports/metricas_modelo.json")


def cargar_metricas():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_modelo_exportado_existe():
    assert MODEL_PATH.exists()


def test_metricas_modelo_existen():
    assert METRICS_PATH.exists()


def test_modelo_seleccionado_valido():
    metricas = cargar_metricas()
    modelos_validos = {
        "LogisticRegression",
        "LinearSVC",
        "KNeighborsClassifier",
        "DecisionTreeClassifier",
    }
    assert metricas["selected_model"] in modelos_validos


def test_modelo_mantiene_f1_minimo():
    metricas = cargar_metricas()
    test_f1 = metricas["metrics"]["test_f1"]
    assert test_f1 >= 0.60


def test_modelo_mantiene_recall_minimo():
    metricas = cargar_metricas()
    test_recall = metricas["metrics"]["test_recall"]
    assert test_recall >= 0.80


def test_modelo_mantiene_cv_f1_minimo():
    metricas = cargar_metricas()
    cv_f1_mean = metricas["metrics"]["cv_f1_mean"]
    assert cv_f1_mean >= 0.80


def test_dataset_modelado_coherente():
    metricas = cargar_metricas()
    assert metricas["dataset_rows"] == 10127
    assert metricas["dataset_columns"] == 36
    assert metricas["target"] == "falta_pago"