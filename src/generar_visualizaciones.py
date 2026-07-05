import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Rutas del proyecto
DATA_RAW = Path("data/raw")
REPORTS_FIGURES = Path("reports/figures")
REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)

# Configuración visual
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 5)


def cargar_datos():
    """Carga los datasets originales del proyecto."""
    df_creditos = pd.read_csv(DATA_RAW / "datos_creditos.csv", sep=";")
    df_tarjetas = pd.read_csv(DATA_RAW / "datos_tarjetas.csv", sep=";")
    return df_creditos, df_tarjetas


def guardar_distribucion_target(df_creditos):
    """Genera y guarda la distribución de la variable objetivo falta_pago."""
    plt.figure()
    sns.countplot(x="falta_pago", data=df_creditos)
    plt.title("Distribución de la variable objetivo: falta_pago")
    plt.xlabel("Falta de pago")
    plt.ylabel("Cantidad de clientes")
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES / "distribucion_variable_objetivo.png")
    plt.close()


def guardar_distribuciones_categoricas(df, nombre_dataset, excluir=None):
    """Genera gráficos de distribución para variables categóricas."""
    excluir = excluir or []
    columnas_categoricas = df.select_dtypes(include=["object"]).columns

    for col in columnas_categoricas:
        if col in excluir:
            continue

        plt.figure(figsize=(8, 4))
        orden = df[col].value_counts().index
        sns.countplot(y=col, data=df, order=orden)
        plt.title(f"Distribución de {col} - {nombre_dataset}")
        plt.xlabel("Cantidad")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(REPORTS_FIGURES / f"distribucion_{nombre_dataset}_{col}.png")
        plt.close()


def guardar_matriz_correlacion(df, nombre_dataset):
    """Genera y guarda una matriz de correlación para variables numéricas."""
    df_numerico = df.select_dtypes(include=["float64", "int64"])

    plt.figure(figsize=(10, 8))
    correlacion = df_numerico.corr()
    sns.heatmap(correlacion, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Matriz de correlaciones - {nombre_dataset}")
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES / f"matriz_correlacion_{nombre_dataset}.png")
    plt.close()


def main():
    df_creditos, df_tarjetas = cargar_datos()

    print("Datos cargados correctamente.")
    print(f"Dataset créditos: {df_creditos.shape[0]} filas y {df_creditos.shape[1]} columnas.")
    print(f"Dataset tarjetas: {df_tarjetas.shape[0]} filas y {df_tarjetas.shape[1]} columnas.")

    guardar_distribucion_target(df_creditos)
    guardar_distribuciones_categoricas(df_creditos, "creditos", excluir=["falta_pago"])
    guardar_distribuciones_categoricas(df_tarjetas, "tarjetas")
    guardar_matriz_correlacion(df_creditos, "creditos")
    guardar_matriz_correlacion(df_tarjetas, "tarjetas")

    print(f"Visualizaciones guardadas en: {REPORTS_FIGURES}")


if __name__ == "__main__":
    main()
