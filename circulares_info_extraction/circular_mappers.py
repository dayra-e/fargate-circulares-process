from circulares_info_extraction.circular_validation import (check_date_format,
                                                                check_number_range, check_identificador_asfi,
                                                                check_nombre_completo, check_numero_carta_circular,
                                                                check_val_in_column, generic_text_validation,
                                                                is_row_value_empty, check_abreviacion_departamento,
                                                                check_razon_social_full_name,
                                                                MapperCircularValidationBuilder)
from circulares_info_extraction.config import LoadConfig
from circulares_info_extraction.circular_validation import apply_validation
from pandas import DataFrame
from pandas.io.formats.style import Styler

config = LoadConfig()
valid_cities = set(config.get_section("ciudades_departamentos_abreviaciones").keys())
valid_cities_abrev = set(config.get_section("ciudades_departamentos_abreviaciones").values())
valid_documentos_identidad = ["CI", "NIT", "RUC"]
valid_moneda = ["BS", "USD", "UFV"]
valid_respaldo = ["NUREJ", "PIET", "IANUS"]

config.set_section("paths")
JUZGADOS_FILE = config.parameter("juzgados_file")
DB_NAMES_FILE = config.parameter("db_names_file")

builder = MapperCircularValidationBuilder()


def build_mapper_ret_sus_rem() -> dict:
    builder.reset()
    builder.add(("FECHA DE CITE", "FECHA DE PUBLICACION DE LA CARTA CIRCULAR"), check_date_format)
    builder.add("NOMBRE DE LA CIUDAD DEL SOLICITANTE", generic_text_validation, valid_values=valid_cities)
    builder.add('NUMERO DE CARTA CIRCULAR', check_numero_carta_circular)
    builder.add("TIPO DE DOCUMENTO DE IDENTIDAD", generic_text_validation,
                valid_values=valid_documentos_identidad)
    builder.add("NUMERO DE DOCUMENTO DE IDENTIDAD", check_number_range, low=100000, high=100000000000)
    builder.add(("TIPO DE DOCUMENTO DE IDENTIDAD", "ABREVIACION DEL DEPARTAMENTO"),
                check_abreviacion_departamento, valid_values=valid_cities_abrev, axis=None)
    builder.add("MONEDA", generic_text_validation, valid_values=valid_moneda)
    builder.add("MONTO A SER RETENIDO", check_number_range, low=0, high=50000000)
    builder.add("IDENTIFICADOR UNICO QUE REGISTRA ASFI", check_identificador_asfi)
    builder.add("DOCUMENTO DE RESPALDO", generic_text_validation, valid_values=valid_respaldo)
    builder.add("NOMBRE DE LA AUTORIDAD SOLICITANTE", check_val_in_column,
                db_path=JUZGADOS_FILE, db_col='NOMBRE')
    builder.add("NOMBRE DEL JUZGADO", check_val_in_column,
                db_path=JUZGADOS_FILE, db_col='AUTORIDAD')
    builder.add(("TIPO DE PROCESO", "NUMERO DE CITE",
                 "DEMANDANTE", "NUMERO TIPO DE RESPALDO"),
                is_row_value_empty)
    builder.add(("RAZON SOCIAL", "APELLIDO PATERNO", "APELLIDO MATERNO", "NOMBRES"),
                check_razon_social_full_name, db_path=DB_NAMES_FILE, axis=1)
    return builder.get_mapper()


def build_mapper_informativa() -> dict:
    builder.reset()
    builder.add(("FECHA DE PUBLICACION DE LA CARTA CIRCULAR", "FECHA DE PUBLICACION"),
                check_date_format)
    builder.add("NOMBRE DE LA CIUDAD DEL SOLICITANTE", generic_text_validation, valid_values=valid_cities)
    builder.add('NUMERO DE CARTA CIRCULAR', check_numero_carta_circular)
    builder.add("IDENTIFICADOR UNICO QUE REGISTRA ASFI", check_identificador_asfi)
    builder.add("NOMBRE DE LA AUTORIDAD SOLICITANTE", check_val_in_column,
                db_path=JUZGADOS_FILE, db_col='NOMBRE')
    builder.add("TIPO DE CASO", generic_text_validation, valid_values=valid_respaldo)
    builder.add("NOMBRE DEL JUZGADO", check_val_in_column,
                db_path=JUZGADOS_FILE, db_col='AUTORIDAD')
    builder.add(("DEMANDANTES", "DEMANDADOS"), check_nombre_completo,
                db_path=DB_NAMES_FILE)
    builder.add(("TIPO DE PROCESO", "NUMERO DOCUMENTO DE RESPALDO"),
                is_row_value_empty)
    return builder.get_mapper()

def build_mapper_combinada() -> dict:
    builder.reset()
    builder.add(("FECHA DE CITE", "FECHA DE PUBLICACION DE LA CARTA CIRCULAR", "FECHA DE PUBLICACION"),
                check_date_format)
    builder.add("NOMBRE DE LA CIUDAD DEL SOLICITANTE", generic_text_validation, valid_values=valid_cities)
    builder.add('NUMERO DE CARTA CIRCULAR', check_numero_carta_circular)
    builder.add("DEMANDADOS", check_nombre_completo, db_path=DB_NAMES_FILE)
    builder.add("TIPO DE DOCUMENTO DE IDENTIDAD", generic_text_validation,
                valid_values=valid_documentos_identidad)
    builder.add("TIPO DE CASO", generic_text_validation, valid_values=valid_respaldo)
    builder.add("NUMERO DE DOCUMENTO DE IDENTIDAD", check_number_range, low=100000, high=100000000000)
    builder.add("MONEDA", generic_text_validation, valid_values=valid_moneda)
    builder.add("MONTO A SER RETENIDO", check_number_range, low=0, high=50000000)
    builder.add("IDENTIFICADOR UNICO QUE REGISTRA ASFI", check_identificador_asfi)
    builder.add("DOCUMENTO DE RESPALDO", generic_text_validation, valid_values=valid_respaldo)
    builder.add("NOMBRE DE LA AUTORIDAD SOLICITANTE", check_val_in_column,
                db_path=JUZGADOS_FILE, db_col='NOMBRE')
    builder.add("NOMBRE DEL JUZGADO", check_val_in_column,
                db_path=JUZGADOS_FILE, db_col='AUTORIDAD')
    builder.add(("TIPO DE PROCESO", "NUMERO DE CITE",
                 "DEMANDANTE", "DEMANDANTES", "NUMERO DOCUMENTO DE RESPALDO"),
                is_row_value_empty)
    builder.add(("RAZON SOCIAL", "APELLIDO PATERNO", "APELLIDO MATERNO", "NOMBRES"),
                check_razon_social_full_name, db_path=DB_NAMES_FILE, axis=1)
    return builder.get_mapper()


def build_mapper_normativa():
    builder.reset()
    builder.add("FECHA DE PUBLICACION DE LA CARTA CIRCULAR", check_date_format)
    builder.add('NUMERO DE CARTA CIRCULAR', check_numero_carta_circular)
    builder.add("NUMERO DE TRAMITE", is_row_value_empty)
    return builder.get_mapper()


def validate_dataframe(tipo_circular: str, dataframe: DataFrame) -> Styler:
    if tipo_circular == "retencion_suspension_remision":
        mapper = build_mapper_ret_sus_rem()
    elif tipo_circular == "informativa":
        mapper = build_mapper_informativa()
    elif tipo_circular == "combinada":
        mapper = build_mapper_combinada()
    elif tipo_circular == 'normativa':
        mapper = build_mapper_normativa()
    else:
        raise TypeError(f"CIRCULAR:: tipo_circular={tipo_circular} not found")
    return apply_validation(dataframe, mapper)
