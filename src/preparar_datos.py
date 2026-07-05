import pandas as pd
from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
REPORTS = Path("reports")

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def cargar_datos():
    creditos = pd.read_csv(DATA_RAW / "datos_creditos.csv", sep=";")
    tarjetas = pd.read_csv(DATA_RAW / "datos_tarjetas.csv", sep=";")
    return creditos, tarjetas


def limpiar_creditos(df):
    df = df.copy()

    # Corrección de tipos
    df["id_cliente"] = df["id_cliente"].astype("int64")

    # Imputación de valores faltantes numéricos
    df["antiguedad_empleado"] = df["antiguedad_empleado"].fillna(
        df["antiguedad_empleado"].median()
    )
    df["tasa_interes"] = df["tasa_interes"].fillna(
        df["tasa_interes"].median()
    )

    # Conversión de variable objetivo
    df["falta_pago"] = df["falta_pago"].map({"Y": 1, "N": 0})

    return df


def limpiar_tarjetas(df):
    df = df.copy()

    # Corrección de tipos
    df["id_cliente"] = df["id_cliente"].astype("int64")

    return df


def integrar_datos(creditos, tarjetas):
    df = creditos.merge(
        tarjetas,
        on="id_cliente",
        how="inner",
        validate="one_to_one"
    )
    return df


def transformar_variables_categoricas(df):
    columnas_categoricas = df.select_dtypes(include=["object"]).columns.tolist()

    df_transformado = pd.get_dummies(
        df,
        columns=columnas_categoricas,
        drop_first=True
    )

    return df_transformado


def generar_reporte(df_original, df_final):
    reporte = REPORTS / "preparacion_datos_resumen.txt"

    with open(reporte, "w", encoding="utf-8") as f:
        f.write("Resumen de preparación de datos\n")
        f.write("================================\n\n")
        f.write(f"Filas iniciales integradas: {df_original.shape[0]}\n")
        f.write(f"Columnas iniciales integradas: {df_original.shape[1]}\n")
        f.write(f"Filas finales: {df_final.shape[0]}\n")
        f.write(f"Columnas finales: {df_final.shape[1]}\n")
        f.write(f"Valores nulos finales: {df_final.isnull().sum().sum()}\n\n")

        f.write("Columnas finales:\n")
        for columna in df_final.columns:
            f.write(f"- {columna}\n")


def main():
    creditos, tarjetas = cargar_datos()

    creditos_limpios = limpiar_creditos(creditos)
    tarjetas_limpios = limpiar_tarjetas(tarjetas)

    datos_integrados = integrar_datos(creditos_limpios, tarjetas_limpios)
    datos_modelo = transformar_variables_categoricas(datos_integrados)

    datos_modelo.to_csv(
        DATA_PROCESSED / "datos_modelo.csv",
        index=False
    )

    generar_reporte(datos_integrados, datos_modelo)

    print("Preparación de datos completada correctamente.")
    print(f"Dataset final: {datos_modelo.shape[0]} filas y {datos_modelo.shape[1]} columnas.")
    print("Archivo generado: data/processed/datos_modelo.csv")
    print("Reporte generado: reports/preparacion_datos_resumen.txt")


if __name__ == "__main__":
    main()