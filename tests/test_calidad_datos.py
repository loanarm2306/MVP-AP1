import pandas as pd
from pathlib import Path

DATA_RAW = Path("data/raw")

COLUMNAS_CREDITOS = [
    "id_cliente",
    "edad",
    "importe_solicitado",
    "duracion_credito",
    "antiguedad_empleado",
    "situacion_vivienda",
    "ingresos",
    "objetivo_credito",
    "pct_ingreso",
    "tasa_interes",
    "estado_credito",
    "falta_pago",
]

COLUMNAS_TARJETAS = [
    "id_cliente",
    "antiguedad_cliente",
    "estado_civil",
    "estado_cliente",
    "gastos_ult_12m",
    "genero",
    "limite_credito_tc",
    "nivel_educativo",
    "nivel_tarjeta",
    "operaciones_ult_12m",
    "personas_a_cargo",
]


def cargar_creditos():
    return pd.read_csv(DATA_RAW / "datos_creditos.csv", sep=";")


def cargar_tarjetas():
    return pd.read_csv(DATA_RAW / "datos_tarjetas.csv", sep=";")


def test_archivos_existen():
    assert (DATA_RAW / "datos_creditos.csv").exists()
    assert (DATA_RAW / "datos_tarjetas.csv").exists()


def test_estructura_columnas_creditos():
    df = cargar_creditos()
    assert list(df.columns) == COLUMNAS_CREDITOS


def test_estructura_columnas_tarjetas():
    df = cargar_tarjetas()
    assert list(df.columns) == COLUMNAS_TARJETAS


def test_misma_cantidad_registros():
    df_creditos = cargar_creditos()
    df_tarjetas = cargar_tarjetas()
    assert len(df_creditos) == len(df_tarjetas)


def test_id_cliente_unico_en_creditos():
    df = cargar_creditos()
    porcentaje_duplicados = df["id_cliente"].duplicated().mean()
    assert porcentaje_duplicados <= 0.10


def test_id_cliente_unico_en_tarjetas():
    df = cargar_tarjetas()
    porcentaje_duplicados = df["id_cliente"].duplicated().mean()
    assert porcentaje_duplicados <= 0.10


def test_integridad_referencial_entre_datasets():
    df_creditos = cargar_creditos()
    df_tarjetas = cargar_tarjetas()

    ids_creditos = set(df_creditos["id_cliente"])
    ids_tarjetas = set(df_tarjetas["id_cliente"])

    ids_sin_tarjeta = ids_creditos - ids_tarjetas
    porcentaje_error = len(ids_sin_tarjeta) / len(ids_creditos)

    assert porcentaje_error <= 0.10


def test_variable_objetivo_valores_validos():
    df = cargar_creditos()
    valores_validos = {"Y", "N"}
    assert set(df["falta_pago"].unique()).issubset(valores_validos)


def test_estado_credito_valores_validos():
    df = cargar_creditos()
    valores_validos = {0, 1}
    assert set(df["estado_credito"].unique()).issubset(valores_validos)


def test_genero_valores_validos():
    df = cargar_tarjetas()
    valores_validos = {"M", "F"}
    assert set(df["genero"].unique()).issubset(valores_validos)


def test_completitud_creditos_umbral_general():
    df = cargar_creditos()
    porcentaje_nulos = df.isnull().sum().sum() / df.size
    assert porcentaje_nulos <= 0.05


def test_completitud_tarjetas_umbral_general():
    df = cargar_tarjetas()
    porcentaje_nulos = df.isnull().sum().sum() / df.size
    assert porcentaje_nulos <= 0.05


def test_rango_edad_creditos():
    df = cargar_creditos()
    errores = df[~df["edad"].between(18, 90)]
    porcentaje_error = len(errores) / len(df)
    assert porcentaje_error <= 0.01


def test_rango_importe_solicitado():
    df = cargar_creditos()
    assert (df["importe_solicitado"] > 0).all()


def test_rango_ingresos():
    df = cargar_creditos()
    assert (df["ingresos"] > 0).all()


def test_rango_limite_credito_tarjeta():
    df = cargar_tarjetas()
    assert (df["limite_credito_tc"] > 0).all()


def test_rango_operaciones_ult_12m():
    df = cargar_tarjetas()
    assert (df["operaciones_ult_12m"] >= 0).all()