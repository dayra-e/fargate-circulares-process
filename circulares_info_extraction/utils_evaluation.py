import re
import pandas as pd
from circulares_info_extraction.config import LoadConfig

config = LoadConfig()
COLS_TO_CLEAN_ACCENTS = config.get_section("cols_to_clean_accents")
CIUDADES_ABREVIACIONES = config.get_section("ciudades_departamentos_abreviaciones")

config.set_section('image_processing')
NOISE_THRESHOLD = config.parameter("noise_threshold")


def check_number_range(val, low, high):
    try:
        num = float(val)
        if low <= num <= high:
            return 'background-color: lightgreen'
    except ValueError:
        pass
    return 'background-color: red'


def check_date_format(val):
    try:
        pd.to_datetime(val, format='%d/%m/%Y', errors='raise')
        return 'background-color: lightgreen'
    except:
        return 'background-color: lightred'


def check_razon_social(val):
    if pd.isna(val):
        return 'background-color: lightyellow'
    return ''


def generic_text_validation(val, valid_values):
    if val in valid_values:
        return 'background-color: lightgreen'
    else:
        return 'background-color: red'


def check_identificador_asfi(val):
    # Define the pattern for the identificador
    pattern = r'^[A-Z]-\d+$'
    if re.match(pattern, val):
        return 'background-color: lightgreen'
    else:
        return 'background-color: red'


def check_numero_carta_circular(val):
    pattern = r'^CC-\d{4}/\d{4}$'
    if re.match(pattern, val):
        return 'background-color: lightgreen'
    else:
        return 'background-color: red'


valid_cities = set(config["ciudades_departamentos_abreviaciones"].keys())
valid_documentos_identidad = ["CI", "NIT", "RUC"]
valid_moneda = ["BS", "USD", "UFV"]
valid_respaldo = ["NUREJ", "PIET", "IANUS"]


def apply_evaluation(df):
    styled_df = df.style.applymap(check_date_format, subset=["FECHA DE CITE",
                                                             "FECHA DE PUBLICACION DE LA CARTA CIRCULAR"]) \
        .applymap(check_numero_carta_circular, subset=['NUMERO DE CARTA CIRCULAR']) \
        .applymap(check_identificador_asfi, subset=["IDENTIFICADOR UNICO QUE REGISTRA ASFI"]) \
        .applymap(check_razon_social, subset=["RAZON SOCIAL"]) \
        .applymap(lambda val: generic_text_validation(val, valid_cities),
                  subset=["NOMBRE DE LA CIUDAD DEL SOLICITANTE"]) \
        .applymap(lambda val: generic_text_validation(val, valid_documentos_identidad),
                  subset=["TIPO DE DOCUMENTO DE IDENTIDAD"]) \
        .applymap(lambda val: check_number_range(val, 100000, 100000000), subset=["NUMERO DE DOCUMENTO DE IDENTIDAD"]) \
        .applymap(lambda val: generic_text_validation(val, valid_moneda), subset=["MONEDA"]) \
        .applymap(lambda val: check_number_range(val, 0, 500000), subset=["MONTO A SER RETENIDO"]) \
        .applymap(lambda val: generic_text_validation(val, valid_respaldo), subset=["DOCUMENTO DE RESPALDO"])
    return styled_df
