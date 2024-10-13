import pandas as pd
from dotenv import load_dotenv

load_dotenv()
import sys

sys.path.append("../src/")
sys.path.append("../../account/")
import os
from fuzzywuzzy import process

from circulares_info_extraction.text_processing import customize_list_with_strings
from circulares_info_extraction.postprocessing import concatenate_if_not_present
from circulares_info_extraction.prompts import prompt_find_juzgado
from circulares_info_extraction.api_clients import openai_client
from circulares_info_extraction.config import LoadConfig

config = LoadConfig()

# Paths
config.set_section('paths')
PATH_TO_JUZGADOS_FILE = config.parameter('juzgados_file')
# Find judges:
config.set_section('find_judges')
MIN_LETTERS_JUDGE = config.parameter("min_num_letters_judge")
JUDGE_NAME_THRESHOLD = config.parameter("nombres_threshold_similarity")
JUZGADO_NAME_THRESHOLD = config.parameter("juzgados_threshold_similarity")


def find_and_update_juzgado_info(i, juzgado_buscar, lista_autoridades, prompt_find_juzgado, llm_model, temperature,
                                 juzgados_df, resultados_df):
    """
    Searches for a juzgado in a list of autoridades using a language model and updates information in a DataFrame.

    Parameters:
    - i: The index of the current row being processed.
    - juzgado_buscar: The specific juzgado to search for.
    - lista_autoridades: List of authorities to search within.
    - prompt_find_juzgado: Customized prompt for the language model.
    - llm_model: The language model to use for generating completions.
    - temperature: The temperature setting for the language model.
    - juzgados_df: DataFrame containing juzgado information.
    - resultados_df: DataFrame to update with the found juzgado information.
    """
    # Customize list with strings and use the language model to find the judge's name
    messages_find_juzgado = customize_list_with_strings(lista_autoridades, juzgado_buscar, prompt_find_juzgado)
    chat_completion = openai_client.chat.completions.create(messages=messages_find_juzgado,
                                                            model=llm_model,
                                                            temperature=temperature)
    resultado = chat_completion.choices[0].message.content.strip("'")

    # Query the juzgados_df DataFrame for a match
    matches_df = juzgados_df.query(f"AUTORIDAD_CIUDAD=='{resultado}'")
    num_matches = matches_df.shape[0]

    if num_matches > 0:
        nombre_juez = matches_df["NOMBRE"].values[0]
        nombre_juzgado = matches_df["AUTORIDAD"].values[0]
        # Update resultados_df with the judge's name and juzgado name
        condition_ciudad_juzgado = (resultados_df["NOMBRE DEL JUZGADO - CIUDAD"] == juzgado_buscar)
        resultados_df.loc[condition_ciudad_juzgado, "NOMBRE DE LA AUTORIDAD SOLICITANTE"] = nombre_juez
        resultados_df.loc[condition_ciudad_juzgado, "NOMBRE DEL JUZGADO"] = nombre_juzgado
        print(f"Found match by juzgado for row {i}: {juzgado_buscar} : {nombre_juzgado}. Juez: {nombre_juez}")
    else:
        print(f"No match in juzgados search for row {i}: {juzgado_buscar}")


def load_juzgados_df(path_to_file):
    """
    Loads a DataFrame from a CSV file containing juzgado information.
    Parameters: path_to_file: String, path to the CSV file.
    Returns:DataFrame: Loaded data from the specified file.
    """
    try:
        df = pd.read_csv(path_to_file)
        df['AUTORIDAD_CIUDAD'] = df.apply(lambda row: concatenate_if_not_present(row, "AUTORIDAD", "CIUDAD"), axis=1)
        df['NOMBRE_CIUDAD'] = df.apply(lambda row: concatenate_if_not_present(row, "NOMBRE", "CIUDAD"), axis=1)
        return df
    except Exception as e:
        print(f"ISSUE reading judges csv. Current path: {os.getcwd()}, error: {e}")
        raise


def process_juzgado_search(i, ciudad, juzgado_buscar, juzgados_df, resultados_df, prompt_find_juzgado, llm_model,
                           temperature):
    """
    Searches for a juzgado using a language model and updates the DataFrame based on the search results.

    Parameters:
    - i: The index of the current row.
    - ciudad: City where the letter comes from
    - juzgado_buscar: The combined name of the juzgado and city to search for.
    - juzgados_df: DataFrame containing information about courts.
    - resultados_df: DataFrame to be updated with the search results.
    - prompt_find_juzgado: The base prompt to be customized for querying the language model.
    - llm_model: The identifier for the language model to be used.
    - temperature: The temperature setting for the language model, affecting its creativity.
    """

    # Reduce the space search by only looking at the juzgados in that city:
    juzgados_in_city = juzgados_df[(juzgados_df["CIUDAD"] == ciudad) | (juzgados_df["AUTORIDAD"].str.contains(ciudad))]
    lista_juzgados_to_look = juzgados_in_city["AUTORIDAD_CIUDAD"].to_list()
    lista_juzgados_to_look_llm = "\n".join(lista_juzgados_to_look)

    # First check using fuzzy matching:
    fuzzy_match_result = find_best_match_with_threshold(juzgado_buscar, lista_juzgados_to_look, JUZGADO_NAME_THRESHOLD)
    if fuzzy_match_result:
        print(f"fuzzy match result: {fuzzy_match_result} for {juzgado_buscar}")
        matches_df = juzgados_df.query(f"AUTORIDAD_CIUDAD=='{fuzzy_match_result[0]}'")
    else:
        print(f"Looking for {juzgado_buscar} with LLM")
        # Look for the juzgado in the reduced df
        messages_find_juzgado = customize_list_with_strings(lista_juzgados_to_look_llm, juzgado_buscar,
                                                            prompt_find_juzgado)
        chat_completion = openai_client.chat.completions.create(messages=messages_find_juzgado, model=llm_model,
                                                                temperature=temperature)
        resultado = chat_completion.choices[0].message.content.strip("'")
        print(f"Result for juzgado search with LLM: {resultado}")
        matches_df = juzgados_df.query(f"AUTORIDAD_CIUDAD=='{resultado}'")
    update_dataframe(matches_df, resultados_df, i, juzgado_buscar, is_juzgado=True)


def find_judge_name(resultados_df, llm_model, temperature):
    """
    Searches and updates the judge's name in a DataFrame based on fuzzy search, this
     not finding a match it uses the provided GPT prompts and a language model.
    Parameters:
    - resultados_df: DataFrame, contains data where the judge's name needs updating.
    - prompt_find_juzgado: String, prompt template for GPT inquiries.
    - llm_model: String, identifier of the language model to use.
    - temperature: Float, temperature setting for the GPT model response variability.
    Returns:
    - None, directly modifies the resultados_df DataFrame.
    """
    print("Loading juzgados csv")
    juzgados_df = load_juzgados_df(PATH_TO_JUZGADOS_FILE)
    lista_nombres = juzgados_df["NOMBRE"].to_list()

    resultados_df["NOMBRE DEL JUZGADO - CIUDAD"] = resultados_df.apply(
        lambda row: concatenate_if_not_present(row, "NOMBRE DEL JUZGADO", "NOMBRE DE LA CIUDAD DEL SOLICITANTE"),
        axis=1)

    ciudad_autoridades_df = resultados_df[["NOMBRE DE LA CIUDAD DEL SOLICITANTE", "NOMBRE DE LA AUTORIDAD SOLICITANTE",
                                           "NOMBRE DEL JUZGADO"]].drop_duplicates().reset_index(drop=True)

    # Inside the find_judge_name function
    for i, row in ciudad_autoridades_df.iterrows():
        nombre = row["NOMBRE DE LA AUTORIDAD SOLICITANTE"]
        juzgado = row["NOMBRE DEL JUZGADO"]
        ciudad = row["NOMBRE DE LA CIUDAD DEL SOLICITANTE"]
        juzgado_buscar = concatenate_if_not_present(row, "NOMBRE DEL JUZGADO", "NOMBRE DE LA CIUDAD DEL SOLICITANTE")
        print(f"\nLooking for row {i}, Judge: {nombre}, from juzgado {juzgado} in {ciudad}")
        if len(nombre) > MIN_LETTERS_JUDGE and "ASFI" not in nombre:
            best_match = find_best_match_with_threshold(nombre, lista_nombres, threshold=JUDGE_NAME_THRESHOLD)
            if best_match:
                process_judge_match(i, best_match[0], juzgados_df, resultados_df, nombre)
            else:
                process_juzgado_search(i, ciudad, juzgado_buscar, juzgados_df, resultados_df, prompt_find_juzgado,
                                       llm_model,
                                       temperature)
        else:
            process_juzgado_search(i, ciudad, juzgado_buscar, juzgados_df, resultados_df, prompt_find_juzgado,
                                   llm_model,
                                   temperature)

    resultados_df.drop(columns=["NOMBRE DEL JUZGADO - CIUDAD"], inplace=True, errors='ignore')


def process_judge_match(i, resultado, juzgados_df, resultados_df, nombre):
    """
    Updates `resultados_df` based on a matching judge's name found in `juzgados_df`.
    Args:
        i (int): Index for the update in `resultados_df`.
        resultado (str): Judge's name to match in `juzgados_df`.
        juzgados_df (pandas.DataFrame): Data source for matching.
        resultados_df (pandas.DataFrame): Data target for updates.
        nombre (str): Context for the update process.

    Description: Locates matches in `juzgados_df` and updates `resultados_df` at the specified index with additional data.
    """
    # print(f"Found match nombre for row {i}: {nombre} :  {resultado}")
    matches_df = juzgados_df.query(f"NOMBRE=='{resultado}'")
    update_dataframe(matches_df, resultados_df, i, nombre)


def update_dataframe(matches_df, resultados_df, i, search_term, is_juzgado=False):
    """
    Updates entries in `resultados_df` based on matches found in `matches_df`.

    Args:
        matches_df (pandas.DataFrame): Dataframe with match results.
        resultados_df (pandas.DataFrame): Dataframe to be updated based on matches.
        i (int): Index of the row being processed, used for logging.
        search_term (str): The term used to find matches, affects update conditions.
        is_juzgado (bool, optional): Flag to determine the type of search; defaults to False.

    Description:
        This function updates the judge's name and court name in `resultados_df` if matches are found in `matches_df`.
        It logs the outcome of the match search, indicating whether a match was found and, if so, details of the matched entry.
    """

    if not matches_df.empty:
        nombre_juez = matches_df["NOMBRE"].iloc[0]
        nombre_juzgado = matches_df["AUTORIDAD"].iloc[0]
        condition = (resultados_df["NOMBRE DEL JUZGADO - CIUDAD"] == search_term) if is_juzgado else (
                resultados_df["NOMBRE DE LA AUTORIDAD SOLICITANTE"] == search_term)
        resultados_df.loc[condition, "NOMBRE DE LA AUTORIDAD SOLICITANTE"] = nombre_juez
        resultados_df.loc[condition, "NOMBRE DEL JUZGADO"] = nombre_juzgado
        if is_juzgado:
            print(
                f"Found match with juzgado search for row {i}: {search_term}. Juzgado : {nombre_juzgado} Juez: {nombre_juez}")
        else:
            print(f"Found match with judge search for row {i}: {search_term}. Juez: {nombre_juez}")
    else:
        print(f"No match found for row {i}: {search_term}")


def find_best_match_with_threshold(nombre, lista_nombres, threshold=75):
    """
    Finds the best match for a given name in a list of names using fuzzy search,
    with a minimum similarity score threshold.

    Parameters:
    - nombre: The name to search for.
    - lista_nombres: A list of names to search within.
    - threshold: The minimum similarity score required for a match (default is 75).

    Returns:
    A tuple containing the best match and its similarity score if the score meets the threshold,
    otherwise None.
    """
    best_match = process.extractOne(nombre, lista_nombres)
    if best_match is not None and best_match[1] >= threshold:
        return best_match
    else:
        return None
