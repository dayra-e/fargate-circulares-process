import unicodedata
import re
import pandas as pd
from circulares_info_extraction.config import LoadConfig
from typing import Union

config = LoadConfig()

# Columns to clean accents:
config.set_section('cols_to_clean_accents')
COLS_TO_CLEAN_ACCENTS_RETENCION_SUSPENSION_REMISION = config.parameter("retencion_suspension_remision")
COLS_TO_CLEAN_ACCENTS_INFORMATIVA = config.parameter("informativa")
COLS_TO_CLEAN_ACCENTS_COMBINADA = config.parameter("combinada")

# Abreviaciones ciudad:
CIUDADES_ABREVIACIONES = config.get_section("ciudades_departamentos_abreviaciones")

# Columns to keep
config.set_section('cols_to_keep')
COLS_RETENCION_SUSPENSION_REMISION = config.parameter("retencion_suspension_remision")
COLS_INFORMATIVA = config.parameter("informativa")
COLS_COMBINADA = config.parameter("combinada")

# Columns info
COLS_INFO = config.get_section('cols_info')
COLS_ORDER_RETENCION_SUSPENSION_REMISION = COLS_INFO + COLS_RETENCION_SUSPENSION_REMISION
COLS_ORDER_INFORMATIVA = COLS_INFO + COLS_INFORMATIVA
COLS_ORDER_COMBINADA = COLS_INFO + COLS_COMBINADA


def is_subset_in_df(df: pd.DataFrame, columns: Union[str, tuple[str, ...]]) -> bool:
    if isinstance(columns, tuple):
        for col in columns:
            if col not in df.columns:
                return False
        return True
    return columns in df.columns


def clean_razon_social_fullname(df: pd.DataFrame) -> pd.DataFrame:
    print("Clean razon social-fullname")
    if is_subset_in_df(df, ('RAZON SOCIAL', 'APELLIDO PATERNO', 'APELLIDO MATERNO', 'NOMBRES')):
        def clean_razon_social(df_row):
            if df_row['RAZON SOCIAL'] != '':
                df_row['APELLIDO PATERNO'] = ''
                df_row['APELLIDO MATERNO'] = ''
                df_row['NOMBRES'] = ''
            return df_row

        return df.apply(clean_razon_social, axis=1)
    return df



def reformat_date_dmy_to_ymd(df: pd.DataFrame) -> pd.DataFrame:
    print("Reformatting date d/m/Y to Y/m/d cell by cell")
    # Iterate over each column
    for column in df.columns:
        if 'FECHA' in column:
            # Apply transformation to each cell individually
            for i, value in df[column].items():
                if pd.notnull(value) and value != "":
                    # Convert the cell value to date format
                    date = pd.to_datetime(value, format='%d/%m/%Y')
                    df.at[i, column] = date.strftime('%Y/%m/%d')
                else:
                    # Handle the error for that particular cell
                    print(f"Error processing date in row {i}, column '{column}': {value}")
                    df.at[i, column] = value  # Optionally leave the original value intact
    return df

def clean_abrev_departamento(df: pd.DataFrame) -> pd.DataFrame:
    print("Clean abreviacion departamento")
    if 'ABREVIACION DEL DEPARTAMENTO' in df:
        def rewrite_abrev_departamento(row_value):
            if row_value == 'CBBA':
                row_value = 'CB'
            if row_value == 'QR':
                row_value = 'OR'
            if row_value == 'PT':
                row_value = 'PO'
            return row_value

        df['ABREVIACION DEL DEPARTAMENTO'] = df['ABREVIACION DEL DEPARTAMENTO'].apply(rewrite_abrev_departamento)
    return df


def prepare_metadata(resultados_df, caso, oficios_inicios_texto):
    resultados_metadata = resultados_df[[
        "FECHA DE PUBLICACION DE LA CARTA CIRCULAR",
        "NUMERO DE CARTA CIRCULAR",
        "IDENTIFICADOR UNICO QUE REGISTRA ASFI",
        "TABLES",
        "OFICIO_TEXTO",
        "OUTPUT_JSON",
        "PROCESSING_TIME",
        "LLM_MODEL",
        "NUM_PAGES",
        "NUM_TOKENS"
    ]].drop_duplicates()
    resultados_metadata["CASO"] = caso
    resultados_metadata["OFICIO_INICIOS"] = oficios_inicios_texto
    print("Metadata prepared and duplicates removed.")
    return resultados_metadata


def clean_resultados(resultados_df, tipo_circular):
    """
    Cleaning some fields and return a new cleaned DataFrame.
    @param resultados_df: Input DataFrame
    @return: A new cleaned DataFrame
    """
    # Create a copy of the DataFrame to avoid modifying the original
    if tipo_circular == "retencion_suspension_remision":
        cols_order = COLS_ORDER_RETENCION_SUSPENSION_REMISION
        cols_to_clean_accents = COLS_TO_CLEAN_ACCENTS_RETENCION_SUSPENSION_REMISION
    elif tipo_circular == "informativa":
        cols_order = COLS_ORDER_INFORMATIVA
        cols_to_clean_accents = COLS_TO_CLEAN_ACCENTS_INFORMATIVA
    elif tipo_circular == "combinada":
        cols_order = COLS_ORDER_COMBINADA
        cols_to_clean_accents = COLS_TO_CLEAN_ACCENTS_COMBINADA
    else:
        print("ERROR with tipo circular")

    df_copy = resultados_df.copy()
    print("Upper case the DF")
    df_copy = df_copy.applymap(lambda x: x.upper() if isinstance(x, str) else x)

    # Clean from accents:
    print("Cleaning accents")
    for col in cols_to_clean_accents:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(clean_accents)

    print("Clean Ciudad")
    if "NOMBRE DE LA CIUDAD DEL SOLICITANTE" in df_copy.columns:
        df_copy["NOMBRE DE LA CIUDAD DEL SOLICITANTE"] = df_copy["NOMBRE DE LA CIUDAD DEL SOLICITANTE"].str.replace(
            "SANTA CRUZ DE LA SIERRA", "SANTA CRUZ", regex=False)
        df_copy["NOMBRE DE LA CIUDAD DEL SOLICITANTE"] = df_copy["NOMBRE DE LA CIUDAD DEL SOLICITANTE"].str.replace(
            "SCZ", "SANTA CRUZ", regex=False)
        df_copy["NOMBRE DE LA CIUDAD DEL SOLICITANTE"] = df_copy["NOMBRE DE LA CIUDAD DEL SOLICITANTE"].str.replace(
            "COCHARAMBA", "COCHABAMBA", regex=False)
        df_copy["NOMBRE DE LA CIUDAD DEL SOLICITANTE"] = df_copy["NOMBRE DE LA CIUDAD DEL SOLICITANTE"].str.replace(
            "SANTISIMA TRINIDAD", "TRINIDAD", regex=False)

    print("Clean tipo de documento de identidad")
    if "TIPO DE DOCUMENTO DE IDENTIDAD" in df_copy.columns:
        df_copy["TIPO DE DOCUMENTO DE IDENTIDAD"] = df_copy["TIPO DE DOCUMENTO DE IDENTIDAD"].apply(
            clean_string_from_non_letters)

    # Check if it is retencion, suspension or remision
    if tipo_circular == "retencion_suspension_remision":
        print("ADD Abreviacion del departamento")
        abreviaciones_posibles = set(CIUDADES_ABREVIACIONES.values())
        abreviaciones_posibles.update(["PT", "CB", "TA", "QR", "CBBA", "SCZ"])
        df_copy["ABREVIACION DEL DEPARTAMENTO"] = df_copy.apply(
            lambda row: fill_abreviacion_departamento(row, abreviaciones_posibles), axis=1)

    print("Clean numero de identidad")
    # Clean numero de identidad:
    if "NUMERO DE DOCUMENTO DE IDENTIDAD" in df_copy.columns:
        df_copy["NUMERO DE DOCUMENTO DE IDENTIDAD"] = df_copy["NUMERO DE DOCUMENTO DE IDENTIDAD"].apply(
            clean_string_from_non_digits)

    print("Clean Nombre del juzgado")
    if "NOMBRE DEL JUZGADO" in df_copy.columns:
        df_copy["NOMBRE DEL JUZGADO"] = df_copy["NOMBRE DEL JUZGADO"].str.replace("N°", "NRO", regex=False)
        df_copy["NOMBRE DEL JUZGADO"] = df_copy["NOMBRE DEL JUZGADO"].str.replace("°", "", regex=False)
        df_copy["NOMBRE DEL JUZGADO"] = df_copy["NOMBRE DEL JUZGADO"].str.replace("º", "", regex=False)
        df_copy["NOMBRE DEL JUZGADO"] = df_copy["NOMBRE DEL JUZGADO"].apply(clean_string)
        df_copy["NOMBRE DEL JUZGADO"] = df_copy["NOMBRE DEL JUZGADO"].apply(clean_accents)
        df_copy["NOMBRE DEL JUZGADO"] = df_copy["NOMBRE DEL JUZGADO"].str.replace("JUZGADO", "JUEZ", regex=False)
        df_copy["NOMBRE DEL JUZGADO"] = df_copy["NOMBRE DEL JUZGADO"].str.replace("FISCALIA", "FISCAL", regex=False)

    print("Clean Numero de cite")
    if "NUMERO DE CITE" in df_copy.columns:
        df_copy["NUMERO DE CITE"] = df_copy["NUMERO DE CITE"].str.replace("N°", "N", regex=False)
        df_copy["NUMERO DE CITE"] = df_copy["NUMERO DE CITE"].str.replace("°", "", regex=False)
        df_copy["NUMERO DE CITE"] = df_copy["NUMERO DE CITE"].str.replace("º", "", regex=False)
        df_copy["NUMERO DE CITE"] = df_copy["NUMERO DE CITE"].apply(clean_string)

    print("Clean monto a ser retenido")
    if "MONTO A SER RETENIDO" in df_copy.columns:
        df_copy["MONTO A SER RETENIDO"] = df_copy["MONTO A SER RETENIDO"].fillna("").astype(str)
        df_copy["MONTO A SER RETENIDO"] = df_copy["MONTO A SER RETENIDO"].str.replace(",", "", regex=False)

    print("Clean moneda")
    if "MONTO A SER RETENIDO" in df_copy.columns:
        df_copy["MONEDA"] = df_copy["MONEDA"].fillna("").astype(str)
        df_copy["MONEDA"] = df_copy["MONEDA"].str.replace("BOLIVIANOS", "BS", regex=False)

    df_copy = clean_razon_social_fullname(df_copy)
    df_copy = reformat_date_dmy_to_ymd(df_copy)
    df_copy = clean_abrev_departamento(df_copy)

    try:
        df_copy = df_copy[cols_order]
    except:
        not_found_cols = list(set(resultados_df.columns).difference(set(cols_order)))
        missing_cols = list(set(cols_order).difference(set(resultados_df.columns)))
        print(f"Problems with columns. Missing: {missing_cols}. Not found: {not_found_cols}")
        pass

    return df_copy


def clean_string_from_non_letters(input_str):
    """
    Removes all non-alphabetical characters from a given string.
    Args: input_str (str or None): The string from which to remove non-alphabetical characters, or None.
    Returns:str: A new string containing only the alphabetical characters from the original string. Returns an empty string if `input_str` is None.
    """
    # Using list comprehension to keep only letters
    if input_str is None or isinstance(input_str, float):
        return ''
    input_str = str(input_str)    
    cleaned_str = ''.join(char for char in input_str if char.isalpha())
    return cleaned_str


def clean_string_from_non_digits(input_str):
    """
    Removes all non-digit characters from a given string.
    Args: input_str (str or None): The string from which to remove non-digit characters, or None.
    Returns:str: A new string containing only the digit characters from the original string. Returns an empty string if `input_str` is None.
    """
    # Using list comprehension to keep only digits
    if input_str is None or isinstance(input_str, float):
        return ''
    input_str = str(input_str)
    cleaned_str = ''.join(char for char in input_str if char.isdigit())
    return cleaned_str


def clean_string(input_string):
    """
    Removes all characters from a string that are outside the Basic Multilingual Plane (BMP).
    The BMP includes Unicode characters from U+0000 to U+FFFF.
    Args: input_string (str or None): The string to be cleaned, or None.
    Returns: str: A string containing only BMP characters or an empty string if `input_string` is None.
    Description: Uses a regular expression to identify and remove all characters that are not part of the BMP.
    """
    # This regex matches characters outside the Basic Multilingual Plane (BMP)
    # The BMP contains characters from U+0000 to U+FFFF
    if input_string is None or isinstance(input_string, float):
        return ''
    else:
        input_string = str(input_string)
        pattern = re.compile('[^\u0000-\uFFFF]', re.UNICODE)
    return pattern.sub('', input_string)


def clean_accents(input_string):
    """
    Removes all accents and other strange characters
    @param input_string:
    @return: cleaned string
    """
    if input_string is None or isinstance(input_string, float):
        return ''
    input_string = str(input_string)
    # Normalize the input string to decompose characters into base characters and accents
    normalized_string = unicodedata.normalize('NFD', input_string)
    # Filter out the characters that are not base characters (i.e., remove accents)
    cleaned_string = ''.join(char for char in normalized_string if unicodedata.category(char) != 'Mn')
    return cleaned_string


def concatenate_if_not_present(row, col1, col2):
    """
    concatenates the row of col1 and col2 if col2 string is not present in col1
    @param row: row
    @return: concatenated strings in new column
    """
    try:
        # Assuming CIUDAD or AUTORIDAD might not be strings and could raise TypeError
        string1 = str(row[col1])  # autoridad
        string2 = str(row[col2])  # Ciudad
        if string2 not in string1:
            return f"{string1} {string2}"
        else:
            return string1
    except TypeError as e:
        # Handle the TypeError specifically or log it
        print(f"Error: {e}")
        # You can return a default value or handle it in another way
        return "Error in concatenation"


def fill_abreviacion_departamento(row, abreviaciones_posibles):
    """
    Determines the department abbreviation from identity document data in a given data row.
    Args:row (dict): A dictionary representing a data row which must include keys for identity document type, document number, and applicant's city name.
                    abreviaciones_posibles (list): A list of possible abbreviations to validate against.
    Returns:str: A valid department abbreviation if found and valid; otherwise, an empty string.
    Description: The function first extracts the type of identity document and cleans the document number from non-alphabetical characters.
        If the document type is 'CI' (Carnet de Identidad), it attempts to determine the department abbreviation either from the document number
        or, if that fails, from a predefined dictionary mapping cities to abbreviations. The result is then validated against a list of
        possible abbreviations to ensure it is acceptable.
    """

    # Extrae el tipo de documento de identidad de la fila actual.
    id_tipo_documento = row["TIPO DE DOCUMENTO DE IDENTIDAD"]

    # Llama a una función no definida aquí, 'clean_string_from_non_letters',
    # para limpiar el número de documento de identidad de caracteres no alfabéticos.
    id_departamento = clean_string_from_non_letters(row["NUMERO DE DOCUMENTO DE IDENTIDAD"])

    # Extrae el nombre de la ciudad del solicitante de la fila actual.
    ciudad = row["NOMBRE DE LA CIUDAD DEL SOLICITANTE"]

    # Verifica si el tipo de documento es un Carnet de Identidad (CI).
    if id_tipo_documento == "CI":
        # Si se extrajo una abreviación de departamento del número de documento,
        # esta se asigna a 'abreviacion_departamento'.
        if id_departamento:
            abreviacion_departamento = id_departamento
        # Si no se encontró una abreviación en el documento, intenta obtenerla
        else:
            abreviacion_departamento = CIUDADES_ABREVIACIONES.get(ciudad, "")
        # Verifica si la abreviación obtenida está dentro de las abreviaciones posibles.
        # Si es así, retorna la abreviación; si no, retorna una cadena vacía.
        if abreviacion_departamento in abreviaciones_posibles:
            return abreviacion_departamento
        else:
            return ""
    # Si el tipo de documento no es CI, retorna directamente una cadena vacía.
    else:
        return ""
