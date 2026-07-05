import json
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATA_PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_PROCESSED / "datos_modelo.csv"
MODEL_PATH = MODELS_DIR / "modelo_mora.pkl"
METRICS_PATH = REPORTS_DIR / "metricas_modelo.json"
EXPERIMENTS_PATH = REPORTS_DIR / "resultados_experimentacion.csv"

TARGET = "falta_pago"
RANDOM_STATE = 42


def cargar_dataset():
    """Carga el dataset preparado en la fase anterior."""
    df = pd.read_csv(DATASET_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"No se encontró la variable objetivo '{TARGET}'.")

    return df


def separar_variables(df):
    """Separa variables predictoras y variable objetivo."""
    columnas_a_eliminar = [TARGET]

    if "id_cliente" in df.columns:
        columnas_a_eliminar.append("id_cliente")

    x = df.drop(columns=columnas_a_eliminar)
    y = df[TARGET]

    return x, y


def aplicar_submuestreo(x_train, y_train):
    """
    Aplica submuestreo de la clase mayoritaria para reducir el desbalance.
    """
    df_train = pd.concat([x_train, y_train.rename(TARGET)], axis=1)

    clase_minoritaria = df_train[df_train[TARGET] == 1]
    clase_mayoritaria = df_train[df_train[TARGET] == 0]

    clase_mayoritaria_sub = clase_mayoritaria.sample(
        n=len(clase_minoritaria),
        random_state=RANDOM_STATE,
    )

    df_balanceado = pd.concat(
        [clase_minoritaria, clase_mayoritaria_sub],
        axis=0,
    ).sample(frac=1, random_state=RANDOM_STATE)

    x_balanceado = df_balanceado.drop(columns=[TARGET])
    y_balanceado = df_balanceado[TARGET]

    return x_balanceado, y_balanceado


def definir_modelos():
    """Define los modelos que serán comparados."""
    modelos = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "LinearSVC": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "KNeighborsClassifier": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=7,
                    ),
                ),
            ]
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=6,
            min_samples_leaf=20,
        ),
    }

    return modelos


def evaluar_modelo(nombre, modelo, x_train, y_train, x_test, y_test):
    """Evalúa un modelo mediante validación cruzada y conjunto de test."""
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = ["accuracy", "precision", "recall", "f1"]

    cv_resultados = cross_validate(
        modelo,
        x_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    modelo.fit(x_train, y_train)
    y_pred = modelo.predict(x_test)

    metricas = {
        "modelo": nombre,
        "cv_accuracy_mean": float(cv_resultados["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_resultados["test_accuracy"].std()),
        "cv_precision_mean": float(cv_resultados["test_precision"].mean()),
        "cv_precision_std": float(cv_resultados["test_precision"].std()),
        "cv_recall_mean": float(cv_resultados["test_recall"].mean()),
        "cv_recall_std": float(cv_resultados["test_recall"].std()),
        "cv_f1_mean": float(cv_resultados["test_f1"].mean()),
        "cv_f1_std": float(cv_resultados["test_f1"].std()),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    return modelo, metricas


def registrar_en_mlflow(nombre, modelo, metricas, train_samples, test_samples):
    """Registra parámetros, métricas y modelo en MLflow."""
    with mlflow.start_run(run_name=nombre):
        mlflow.log_param("modelo", nombre)
        mlflow.log_param("target", TARGET)
        mlflow.log_param("train_samples", train_samples)
        mlflow.log_param("test_samples", test_samples)
        mlflow.log_param("balancing_method", "undersampling")
        mlflow.log_param("random_state", RANDOM_STATE)

        for metrica, valor in metricas.items():
            if metrica != "modelo":
                mlflow.log_metric(metrica, valor)

        mlflow.sklearn.log_model(modelo, artifact_path="model")


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("prediccion_mora_creditos")

    df = cargar_dataset()
    x, y = separar_variables(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    x_train_bal, y_train_bal = aplicar_submuestreo(x_train, y_train)

    modelos = definir_modelos()
    resultados = []
    modelos_entrenados = {}

    print("Inicio de la experimentación de modelos.")
    print(f"Dataset total: {df.shape[0]} filas y {df.shape[1]} columnas.")
    print(f"Entrenamiento balanceado: {x_train_bal.shape[0]} registros.")
    print(f"Test: {x_test.shape[0]} registros.")

    for nombre, modelo in modelos.items():
        modelo_entrenado, metricas = evaluar_modelo(
            nombre,
            modelo,
            x_train_bal,
            y_train_bal,
            x_test,
            y_test,
        )

        registrar_en_mlflow(
            nombre,
            modelo_entrenado,
            metricas,
            train_samples=x_train_bal.shape[0],
            test_samples=x_test.shape[0],
        )

        modelos_entrenados[nombre] = modelo_entrenado
        resultados.append(metricas)

        print(
            f"{nombre} | "
            f"CV F1: {metricas['cv_f1_mean']:.4f} | "
            f"Test F1: {metricas['test_f1']:.4f} | "
            f"Test Recall: {metricas['test_recall']:.4f}"
        )

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(EXPERIMENTS_PATH, index=False)

    mejor_resultado = df_resultados.sort_values(
        by="cv_f1_mean",
        ascending=False,
    ).iloc[0]

    mejor_modelo_nombre = mejor_resultado["modelo"]
    mejor_modelo = modelos_entrenados[mejor_modelo_nombre]

    joblib.dump(mejor_modelo, MODEL_PATH)

    metricas_finales = {
        "selected_model": mejor_modelo_nombre,
        "target": TARGET,
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "train_samples_balanced": int(x_train_bal.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "metrics": mejor_resultado.to_dict(),
        "feature_columns": x.columns.tolist(),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metricas_finales, f, indent=4, ensure_ascii=False)

    print("\nEntrenamiento completado correctamente.")
    print(f"Modelo seleccionado: {mejor_modelo_nombre}")
    print(f"Modelo exportado en: {MODEL_PATH}")
    print(f"Métricas exportadas en: {METRICS_PATH}")
    print(f"Resultados de experimentación en: {EXPERIMENTS_PATH}")


if __name__ == "__main__":
    main()