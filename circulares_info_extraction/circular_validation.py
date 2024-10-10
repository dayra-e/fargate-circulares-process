import re
import pandas as pd
from numpy import dtype
from pandas.io.formats.style import Styler
from typing import Tuple, Union, Callable, Dict
from api.circulares_info_extraction.utils_etl import meausure_execution_time


def check_number_range(val, low, high):
    try:
        num = float(val)
        if low <= num <= high:
            return 'background-color: lightgreen'
    except ValueError:
        pass
    return 'background-color: lightcoral'


def check_date_format(val):
    try:
        pd.to_datetime(val, format='%Y/%m/%d', errors='raise')
        return 'background-color: lightgreen'
    except:
        return 'background-color: lightcoral'


def is_value_empty(val):
    return pd.isna(val) or val == ''


def is_row_value_empty(val):
    if is_value_empty(val):
        return 'background-color: lightcoral'
    return 'background-color: lightgreen'


def generic_text_validation(val, valid_values):
    if val in valid_values:
        return 'background-color: lightgreen'
    else:
        return 'background-color: lightcoral'


def check_identificador_asfi(val):
    # Define the pattern for the identificador
    pattern = r'^[A-Z]-\d+$'
    if re.match(pattern, val):
        return 'background-color: lightgreen'
    else:
        return 'background-color: lightcoral'


def check_numero_carta_circular(val):
    pattern = r'^CC-\d{1,8}/\d{4}$'
    if re.match(pattern, val):
        return 'background-color: lightgreen'
    else:
        return 'background-color: lightcoral'


def check_val_in_column(val, db_path, db_col):
    _db = pd.read_csv(db_path)
    if len(_db.loc[val == _db[db_col]]) > 0:
        return 'background-color: lightgreen'
    return 'background-color: lightcoral'


def __from_db_find_name(db_path: str) -> iter:
    with open(db_path, "r") as f:
        for line in f.readlines():
            if not line.startswith("#"):
                yield line.split()[0]


def check_nombre_completo(val, db_path):
    full_name = [x.capitalize() for x in val.split()]
    name_size = len(full_name)
    for db_name in __from_db_find_name(db_path):
        if db_name in full_name:
            full_name.remove(db_name)
            # print(f"check_nombre:pass: {db_name} | {full_name}")
        if len(full_name) == 0:
            return 'background-color: lightgreen'
    return 'background-color: yellow' if len(full_name) < name_size else 'background-color: lightcoral'


def check_abreviacion_departamento(df, valid_values):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    rows_with_ci = df["TIPO DE DOCUMENTO DE IDENTIDAD"] == 'CI'
    valid_cities = df["ABREVIACION DEL DEPARTAMENTO"].isin(valid_values)
    styles.loc[rows_with_ci, "ABREVIACION DEL DEPARTAMENTO"] = 'background-color: lightcoral'
    styles.loc[rows_with_ci & valid_cities, "ABREVIACION DEL DEPARTAMENTO"] = 'background-color: lightgreen'
    return styles


def check_razon_social_full_name(df_row, db_path):
    row_style = {
        'APELLIDO PATERNO': '',
        'APELLIDO MATERNO': '',
        'NOMBRES': '',
        'RAZON SOCIAL': ''
    }
    if is_value_empty(df_row['RAZON SOCIAL']):
        row_style['APELLIDO PATERNO'] = is_row_value_empty(df_row['APELLIDO PATERNO'])
        row_style['APELLIDO MATERNO'] = is_row_value_empty(df_row['APELLIDO MATERNO'])
        row_style['NOMBRES'] = check_nombre_completo(df_row['NOMBRES'], db_path)
    else:
        row_style['RAZON SOCIAL'] = 'background-color: lightgreen'
    return pd.Series(row_style)


class MapperCircularValidationBuilder:
    def __init__(self):
        self.__mapper = {}

    def add(self, keys: Union[str, Tuple[str, ...]], function: Callable, **kwargs):
        """
        Apply a CSS-styling function elementwise, column-wise, row-wise, or table-wise.

        :param keys: A valid 2d input to DataFrame.loc[<subset>]
        :type keys: label, tuple
        :param function:
            Should take a scalar and return a string.
            Should take a Series if ``axis`` in kwargs is in [0,1,None] and return a list-like.
        :type function: function
        :param kwargs:
            Pass along to ``function``.
            If ``axis={None, 0, 1}`` in kwargs uses Styler.apply instead of Styler.applymap
        :return: None

        .. seealso:: pandas.io.formats.style.Styler.apply, pandas.io.formats.style.Styler.applymap version < 2.1
        """

        def validate():
            if not isinstance(function, Callable):
                raise TypeError(f"MAPPERBUILDER:: {function} must be callable")
            if kwargs is not None and not isinstance(kwargs, dict):
                raise TypeError(f"MAPPERBUILDER:: {kwargs} must be dict")

        validate()
        self.__mapper[keys] = {'f': function, 'kwargs': kwargs}

    def get_mapper(self) -> Dict:
        return self.__mapper.copy()

    def reset(self):
        self.__mapper.clear()


def check_columns_in_df(dataframe, columns):
    for column in columns:
        if isinstance(column, tuple):
            for col in column:
                if not col in dataframe:
                    raise KeyError(f"apply_validation: column {col} not in DF")
        else:
            if not column in dataframe:
                raise KeyError(f"apply_validation: column {column} not in DF")


@meausure_execution_time()
def apply_validation(df: pd.DataFrame, mapper: dict) -> Styler:
    check_columns_in_df(df, mapper.keys())
    # Apply evaluations on the class pandas.io.formats.style.Styler
    df_style = df.style

    # Iterate over dict mapper, where keys are DataFrame columns, and values are functions to apply
    for columns, mapping in mapper.items():
        if isinstance(columns, tuple):
            columns = list(columns)
        if mapping['kwargs'] is None:
            df_style = df_style.applymap(mapping['f'], subset=columns)
        else:
            if 'axis' in mapping['kwargs']:
                df_style = df_style.apply(mapping['f'], subset=columns, **mapping['kwargs'])
            else:
                df_style = df_style.applymap(mapping['f'], subset=columns, **mapping['kwargs'])
    return df_style


if __name__ == '__main__':
    from pprint import pprint


    def f2(col, a, b):
        print(a, b)
        return 'background-color: lightred'


    f1 = lambda col: 'background-color: lightgreen'
    df = pd.DataFrame([{"Moneda": '200', 'Tipo': 'oficio'}])
    print("A" in df)
    mbuilder = MapperCircularValidationBuilder()
    mbuilder.add("Moneda", f1)
    mbuilder.add("Tipo", f2, a=10, b=20)
    pprint(mbuilder.get_mapper())
    print(apply_validation(df, mbuilder.get_mapper()))
