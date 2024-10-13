import pandas as pd
from time import time
import tiktoken
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random
from concurrent.futures import ThreadPoolExecutor, as_completed
from circulares_info_extraction.utils_etl import (harmonize_dataframe,
                                                      flatten_dictionary,
                                                      get_todays_date_in_spanish)

from circulares_info_extraction.process_tables import (check_tables,
                                                           extract_tables_from_images_to_md,
                                                           extract_tables_from_images_to_md_parallel,
                                                           split_markdown_into_batches,
                                                           markdown_to_dataframe)


from circulares_info_extraction.api_clients import textract_client

from circulares_info_extraction.text_anonymization import Anonymizer
from circulares_info_extraction.utils_etl import parallel_process
from circulares_info_extraction.parsers import parse_to_text, parse_to_json, parse_markdown, parse_dict_keys
from circulares_info_extraction.config import LoadConfig
from circulares_info_extraction.prompt_models import CircularStandardModel, CircularFirstPageModel, CircularTablaInfoProcesoModel

config = LoadConfig()

# Anonymization
config.set_section('anonymization')
ANONYMIZATION_SWITCH = config.parameter('switch')
print(f"Using anonymization: {ANONYMIZATION_SWITCH}")

# Open AI
config.set_section('openai')
STRUCTURED_OUT_FIRST_PAGE = config.parameter('first_page')
STRUCTURED_OUT_STANDARD = config.parameter('standard')
STRUCTURED_OUT_INFO_PROCESO = config.parameter('tabla_info_proceso')
CIRCULAR_FIRST_PAGE_MODEL = None
CIRCULAR_STANDARD_MODEL = None
CIRCULAR_TABLA_INFO_PROCESO_MODEL = None
if STRUCTURED_OUT_FIRST_PAGE:
    from circulares_info_extraction.prompts_structured import prompt_extraccion_info_oficios

    CIRCULAR_FIRST_PAGE_MODEL = CircularFirstPageModel
else:
    from circulares_info_extraction.prompts import prompt_extraccion_info_oficios

if ANONYMIZATION_SWITCH:
    from circulares_info_extraction.prompts_with_anonymization import (prompt_extraction_table,
                                                        prompt_informativas,
                                                        prompt_clasificar_oficio,
                                                        prompt_extraction_normativa)

    if STRUCTURED_OUT_STANDARD:
        from circulares_info_extraction.prompts_structured_with_anonymization import prompt_extraction_standard

        CIRCULAR_STANDARD_MODEL = CircularStandardModel
    else:
        from circulares_info_extraction.prompts_with_anonymization import prompt_extraction_standard

    if STRUCTURED_OUT_INFO_PROCESO:
        from circulares_info_extraction.prompts_structured_with_anonymization import prompt_extraction_tabla_info_proceso

        CIRCULAR_TABLA_INFO_PROCESO_MODEL = CircularTablaInfoProcesoModel
    else:
        from circulares_info_extraction.prompts_with_anonymization import prompt_extraction_tabla_info_proceso

    anonymizer = Anonymizer()
else:
    from circulares_info_extraction.prompts import (prompt_extraction_table,
                                                        prompt_informativas,
                                                        prompt_clasificar_oficio,
                                                        prompt_extraction_normativa)

    if STRUCTURED_OUT_STANDARD:
        from circulares_info_extraction.prompts_structured import prompt_extraction_standard
        CIRCULAR_STANDARD_MODEL = CircularStandardModel
    else:
        from circulares_info_extraction.prompts import prompt_extraction_standard

    if STRUCTURED_OUT_INFO_PROCESO:
        from circulares_info_extraction.prompts_structured import prompt_extraction_tabla_info_proceso

        CIRCULAR_TABLA_INFO_PROCESO_MODEL = CircularTablaInfoProcesoModel
    else:
        from circulares_info_extraction.prompts import prompt_extraction_tabla_info_proceso



# Open AI
model_128k = config.parameter('model_128k')
TEMPERATURE = config.parameter('temperature')
config_tokens = config.parameter('tokens')
MAX_TOKENS_GPT_4o = config.parameter('max_gpt_4o')
LINES_PER_BATCH = config.parameter('lines_per_batch')
MAX_BATCHES_TABLES = config.parameter('max_batches_tables')



# Tables
config.set_section('tables')
NUM_COLS_TABLE = config.parameter("num_cols_table")
# Concurrecy
config.set_section('concurrency')
MAX_WORKERS = config.parameter('max_workers')
MAX_WORKERS_GPT = config.parameter("max_workers_gpt")

# Columns to keep
config.set_section('cols_to_keep')
COLS_RETENCION_SUSPENSION_REMISION = config.parameter("retencion_suspension_remision")
COLS_RETENCION_SUSPENSION_REMISION_NO_CI_ABREVIACION = [i for i in COLS_RETENCION_SUSPENSION_REMISION if
                                                        i != "ABREVIACION DEL DEPARTAMENTO"]
# Columns info
COLS_INFO = config.get_section('cols_info')
# Other constants
EMPTY_JSON_RETENCION_SUSPENSION_REMISION = {i: None for i in COLS_RETENCION_SUSPENSION_REMISION_NO_CI_ABREVIACION}
# Cols informativa
COLS_INFORMATIVA = config.parameter("informativa")
EMPTY_JSON_INFORMATIVA = {i: None for i in COLS_INFORMATIVA}
# Cols normativa
COLS_NORMATIVA = config.parameter("normativa")

def customize_list_with_string(input_string, parameterized_list):
    """
    Customizes a list of dictionaries, parameterizing one of the values based on an input string.

    Parameters:
    - input_string (str): The string value to be used for parameterization.
    - parameterized_list (list of dicts): The list to be customized, where one or more values
      will be parameterized based on the input string.

    Returns:
    - customized_list (list of dicts): The customized list with values parameterized as specified.
    """

    # Clone the original list to avoid modifying it directly
    customized_list = [dict(item) for item in parameterized_list]

    # Iterate through the list and modify the specified value based on the input string
    for item in customized_list:
        # Example of customizing a specific key with the input string
        # This assumes there's a key in your dictionaries that should contain the parameterized content
        if 'content' in item:  # Check if the key exists
            item['content'] = item['content'].format(input_string=input_string)

    return customized_list


def customize_list_with_strings(input_string1, input_string2, parameterized_list):
    """
    Customizes a list of dictionaries, parameterizing two of the values based on input strings.

    Parameters:
    - input_string1 (str): The first string value to be used for parameterization.
    - input_string2 (str): The second string value to be used for parameterization.
    - parameterized_list (list of dicts): The list to be customized, where one or more values
      will be parameterized based on the input strings.

    Returns:
    - customized_list (list of dicts): The customized list with values parameterized as specified.
    """

    # Clone the original list to avoid modifying it directly
    customized_list = [dict(item) for item in parameterized_list]

    # Iterate through the list and modify the specified values based on the input strings
    for item in customized_list:
        # Example of customizing specific keys with the input strings
        # This assumes there are keys in your dictionaries that should contain the parameterized content
        if 'content' in item:  # Check if the key exists
            item['content'] = item['content'].format(input_string1=input_string1, input_string2=input_string2)
    return customized_list


def adjust_text_length(oficio, message, prompt_output, llm_model, MAX_TOKENS=MAX_TOKENS_GPT_4o):
    """
    Adjusts the length of 'oficio' to ensure the total token count does not exceed MAX_TOKENS.

    :param oficio: The original text whose length may need to be adjusted.
    :param message: A list of message dictionaries for message extraction.
    :param prompt: A list of message dictionaries for prompt extraction.
    :param llm_model: The model name to be used for token counting.
    :param MAX_TOKENS: The maximum allowed number of tokens.
    :return: A tuple containing the new length of 'oficio' and the adjusted 'oficio' text.
    """
    # Compute total number of tokens from the message and prompt
    total_num_tokens = num_tokens_from_messages(message, llm_model) + num_tokens_from_messages(prompt_output, llm_model)

    # Check if the total number of tokens exceeds the maximum allowed
    if total_num_tokens > MAX_TOKENS:
        print(f"Adjusting the length. Current number of tokens:  {total_num_tokens}")
        # Calculate the proportion of the text to keep based on token counts
        proportion_to_keep = MAX_TOKENS / total_num_tokens
        new_length = int(len(oficio) * proportion_to_keep)
        new_oficio = oficio[:new_length]
        num_words = len(oficio.split())
        num_words_new = len(new_oficio.split())
        print(f"Old oficio,  num words : {num_words}. New oficio, num words : {num_words_new}")
    else:
        # No adjustment needed
        new_length = len(oficio)
        new_oficio = oficio

    return new_length, new_oficio


def num_tokens_from_messages(messages, model=model_128k):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-0125-preview"
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301" or model == "gpt-3.5-turbo-0125":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

@retry(stop=stop_after_attempt(10), wait=wait_fixed(2) + wait_random(0, 2))
def extraer_json_info_oficios(info_oficios, client, llm_model, temperature=0):
    """
    Extrae la information de la pagina inicial info_oficios
    """
    # Sacar el numero de oficios

    mensaje_extraccion_info_oficios = customize_list_with_string(info_oficios,
                                                                 prompt_extraccion_info_oficios)

    resultado_info_json = extract_and_parse_to_json(mensaje_extraccion_info_oficios,
                                                    client,
                                                    llm_model,
                                                    temperature,
                                                    prompt_model=CIRCULAR_FIRST_PAGE_MODEL)
    return resultado_info_json


@retry(stop=stop_after_attempt(10), wait=wait_fixed(2) + wait_random(0, 2))
def extract_and_parse_to_text(messages, client, llm_model, temperature):
    """
    Attempts to extract information and parse it to JSON, retrying up to 10 times with a fixed wait of 1 second between tries.

    Parameters:
    - messages_extract (list): The sequence of messages for the chat completion.
    - client: The API client instance used for making chat completion requests.
    - llm_model (str): The llm_model identifier used for the chat completion.
    - temperature (float): The temperature setting for the llm_model's response.

    Returns:
    - A JSON object with the extracted information, or None if parsing fails.
    """
    print(f"Completing and parsing to Json")
    chat_completion = client.chat.completions.create(messages=messages,
                                                     model=llm_model,
                                                     temperature=temperature)
    resultados = chat_completion.choices[0].message.content
    print(f"Resultados: {resultados}")  # For debugging or logging

    resultados = parse_to_text(resultados)
    if resultados is None:
        raise ValueError("Failed to parse JSON")

    return resultados


@retry(stop=stop_after_attempt(10), wait=wait_fixed(2) + wait_random(0, 2))
def extract_and_parse_to_json(messages, client, llm_model, temperature, prompt_model=None):
    """
    Attempts to extract information and parse it to JSON, retrying up to 10 times with a fixed wait of 1 second between tries.

    Parameters:
    - messages_extract (list): The sequence of messages for the chat completion.
    - client: The API client instance used for making chat completion requests.
    - llm_model (str): The llm_model identifier used for the chat completion.
    - temperature (float): The temperature setting for the llm_model's response.
    - prompt_model (pydantic.BaseModel): Model used to parse the llm response.

    Returns:
    - A JSON object with the extracted information, or None if parsing fails.
    """
    print(f"Completing and parsing to Json")
    chat_completion_kwargs = {'messages': messages,
                              'model': llm_model,
                              'temperature': temperature,
                              }
    if prompt_model is not None:
        chat_completion_kwargs.update({'response_format': prompt_model})
        chat_completion = client.beta.chat.completions.parse(**chat_completion_kwargs)
        resultados = chat_completion.choices[0].message.parsed.dict()
        resultados_json = parse_dict_keys(resultados)
        print(resultados_json)
    else:
        chat_completion = client.chat.completions.create(**chat_completion_kwargs)
        resultados = chat_completion.choices[0].message.content
        resultados_json = parse_to_json(resultados)
    print(f"Resultados: {resultados}")  # For debugging or logging
    print(f"Resultados JSON: {resultados_json}")  # For debugging or logging
    if ANONYMIZATION_SWITCH:
        try:
            resultados_json = anonymizer.deanonymize_entities(resultados_json)
            # print(f"Decripted: {resultados_json}")
        except Exception as e:
            print(f"Error: {e}")
    # For debugging or logging
    if resultados_json is None:
        raise ValueError("Failed to parse JSON")

    return resultados_json


@retry(stop=stop_after_attempt(10), wait=wait_fixed(1) + wait_random(0, 1))
def extract_and_parse_to_md(messages, client, llm_model, temperature):
    """
    Attempts to extract information and parse it to JSON, retrying up to 10 times with a fixed wait of 1 second between tries.

    Parameters:
    - messages_extract (list): The sequence of messages for the chat completion.
    - client: The API client instance used for making chat completion requests.
    - llm_model (str): The llm_model identifier used for the chat completion.
    - temperature (float): The temperature setting for the llm_model's response.

    Returns:
    - A JSON object with the extracted information, or None if parsing fails.
    """
    print(f"Completing and parsing to MD")
    chat_completion = client.chat.completions.create(messages=messages,
                                                     model=llm_model,
                                                     temperature=temperature)
    resultados = chat_completion.choices[0].message.content
    resultados_md = parse_markdown(resultados)
    print("Resultados:", resultados)
    if ANONYMIZATION_SWITCH:
        try:
            resultados_md = anonymizer.deanonymize_entities_str(resultados_md)
            # print("Decripted:",resultados_md)
        except Exception as e:
            print(f"Error parsing markdown: {e}")
    if resultados_md is None:
        raise ValueError("Failed to parse to Markdown")

    return resultados_md

# Processing Oficios:
def process_retencion_suspension_remision_in_parallel(oficios,
                                                      imagenes_oficios_separados,
                                                      ids_oficios,
                                                      resultado_info_json,
                                                      client,
                                                      llm_model,
                                                      temperature,
                                                      max_workers=MAX_WORKERS):
    """
    Processes each pair of oficio and imagen_oficio in parallel and concatenates the results into a single DataFrame.
    Orders the resulting DataFrame by "IDENTIFICADOR UNICO QUE REGISTRA ASFI" according to the order in ids_oficios.

    :param oficios: List of oficios.
    :param imagenes_oficios_separados: List of separated office images.
    :param ids_oficios: List of office IDs, defining the order of the result.
    :param resultado_info_json: JSON object with result info.
    :param llm_model: The llm_model used for processing.
    :param temperature: Temperature parameter for the processing function.
    :param max_workers: Maximum number of worker threads to use for parallel processing.
    :return: A concatenated DataFrame of all results, ordered by "IDENTIFICADOR UNICO QUE REGISTRA ASFI".
    """

    def process_oficio(oficio, imagen_oficio, id_oficio):
        # Adapted to process individual items directly
        resultado = extraer_informacion_de_cada_retencion_suspension_remision(resultado_info_json,
                                                                              oficio,
                                                                              imagen_oficio,
                                                                              id_oficio,
                                                                              len(ids_oficios),
                                                                              client,
                                                                              llm_model,
                                                                              temperature,
                                                                              prompt_extraction_standard,
                                                                              prompt_extraction_table,
                                                                              prompt_extraction_tabla_info_proceso,
                                                                              COLS_RETENCION_SUSPENSION_REMISION_NO_CI_ABREVIACION,
                                                                              prompt_model_standard=CIRCULAR_STANDARD_MODEL,
                                                                              prompt_model_info_proceso=CIRCULAR_TABLA_INFO_PROCESO_MODEL
                                                                              )
        return resultado
    results = parallel_process(process_oficio, oficios, imagenes_oficios_separados, ids_oficios,
                               max_workers=MAX_WORKERS)
    # Concatenate all result DataFrames into a single DataFrame and sort
    print(f"Concatenating {len(results)} results resetting the index for secondary sorting")
    # Reset the index of each DataFrame in 'results' and keep the original index as a column
    results_with_index = [df.reset_index() for df in results]
    # Concatenate results with the original index preserved
    final_result_df = pd.concat(results_with_index, ignore_index=True)

    print("Ordering by ASFI ID")
    if len(ids_oficios) > 1:
        final_result_df['sort_order'] = final_result_df['IDENTIFICADOR UNICO QUE REGISTRA ASFI'].apply(
            lambda x: ids_oficios.index(x) if x in ids_oficios else len(ids_oficios)
        )
        # Sort by 'sort_order' and then by the original index to keep the initial order as a secondary criterion
        final_result_df.sort_values(by=['sort_order', 'index'], inplace=True)
        # Drop the 'sort_order' column as it's no longer needed after sorting
        final_result_df.drop(['sort_order', 'index'], axis=1, inplace=True)
        # Optionally, if you want to remove the 'index' column as well, you can drop it or set it as the index
        final_result_df.reset_index(drop=True, inplace=True)  # If you want to drop the 'index' column
    return final_result_df


def extraer_informacion_de_cada_retencion_suspension_remision(resultado_info_json,
                                                              oficio,
                                                              imagen_oficio,
                                                              id_oficio,
                                                              num_oficios,
                                                              client,
                                                              llm_model,
                                                              temperature,
                                                              prompt_extraction_standard,
                                                              prompt_extraction_table,
                                                              prompt_extraction_info_proceso,
                                                              cols_to_keep,
                                                              prompt_model_standard,
                                                              prompt_model_info_proceso):
    """
    Extracts information if each oficio is on a single page:
    """
    print(f"Processing Oficio with ID {id_oficio}")
    print(f"Check if it has tables")

    start_time = time()
    has_tables = check_tables(imagen_oficio)
    if has_tables:
        print(f"Oficio with ID {id_oficio} has tables")
        resultados_proceso_df, oficio_texto, resultados_json, llm_model, num_tokens = process_oficio_with_tables(
            imagen_oficio,
            oficio,
            num_oficios,
            textract_client,
            client,
            llm_model,
            temperature,
            prompt_extraction_table,
            prompt_extraction_info_proceso,
            prompt_model=prompt_model_info_proceso
        )
        if resultados_proceso_df is None:
            print(f"Oficio with ID {id_oficio} was recognised to have tables but no information was extracted, testing without tables")
            resultados_proceso_df, oficio_texto, resultados_json, llm_model, num_tokens = process_oficio_without_tables(
                oficio,
                client,
                llm_model,
                temperature,
                prompt_extraction_standard,
                EMPTY_JSON_RETENCION_SUSPENSION_REMISION,
                prompt_model=prompt_model_standard
            )

    else:
        print(f"Oficio with ID {id_oficio} doesn't have tables")
        resultados_proceso_df, oficio_texto, resultados_json, llm_model, num_tokens = process_oficio_without_tables(
            oficio,
            client,
            llm_model,
            temperature,
            prompt_extraction_standard,
            EMPTY_JSON_RETENCION_SUSPENSION_REMISION,
            prompt_model=prompt_model_standard
        )



    print("Finished processing oficios with LLM")
    # Keep only the required columns
    resultados_proceso_df = harmonize_dataframe(resultados_proceso_df, cols_to_keep)
    print("Dataframe harmonized")
    process_time = time() - start_time

    update_resultados_with_info_and_metadata(resultados_proceso_df, resultado_info_json, id_oficio, has_tables,
                                             oficio_texto, resultados_json, process_time, llm_model, oficio, num_tokens)

    return resultados_proceso_df

def process_informativas_in_parallel(oficios,
                                     ids_oficios,
                                     resultado_info_json,
                                     client,
                                     llm_model,
                                     temperature,
                                     max_workers=MAX_WORKERS):
    """
    Procesa oficios informativos en paralelo y concatena los resultados en un DataFrame final.
    Args:
        oficios (list of list of str): Lista de oficios, cada uno representado como una lista de cadenas de texto.
        ids_oficios (list of str): Lista de identificadores únicos para cada oficio.
        resultado_info_json (dict): Información JSON adicional necesaria para el procesamiento.
        client (object): Cliente para interactuar con el modelo de lenguaje.
        llm_model (str): Nombre del modelo de lenguaje a utilizar.
        temperature (float): Valor de la temperatura para el modelo de lenguaje, que controla la aleatoriedad de las respuestas.
        max_workers (int, optional): Número máximo de hilos a utilizar en el procesamiento paralelo. Por defecto es 5.
    Returns: pandas.DataFrame: DataFrame final que contiene los resultados concatenados y ordenados.
    """

    def process_oficio(oficio, id_oficio):
        """
         Procesa un oficio individual y devuelve el resultado como un DataFrame.
         Args:
             oficio (list of str): El texto del oficio.
             id_oficio (str): El identificador único del oficio.
         Returns: pandas.DataFrame: DataFrame con el resultado del procesamiento del oficio.
         """
        # Adapted to process individual items directly
        resultados = extraer_informacion_de_cada_informativa(resultado_info_json,
                                                             oficio,
                                                             id_oficio,
                                                             client,
                                                             llm_model,
                                                             temperature,
                                                             prompt_informativas,
                                                             COLS_INFORMATIVA)
        return resultados
    results = parallel_process(process_oficio, oficios, ids_oficios, max_workers=max_workers)
    results_with_index = [df.reset_index() for df in results]
    # Concatenate results with the original index preserved
    final_result_df = pd.concat(results_with_index, ignore_index=True)
    return final_result_df


def extraer_informacion_de_cada_informativa(resultado_info_json,
                                            oficio,
                                            id_oficio,
                                            client,
                                            llm_model,
                                            temperature,
                                            prompt_informativas,
                                            cols_to_keep):
    """
    Extracts information if each oficio is on a single page:
    """
    print(f"Processing Oficio with ID {id_oficio}")
    start_time = time()
    has_tables = False
    resultados_proceso_df, oficio_texto, resultados_json, llm_model, num_tokens = process_oficio_without_tables(oficio,
                                                                                                                client,
                                                                                                                llm_model,
                                                                                                                temperature,
                                                                                                                prompt_informativas,
                                                                                                                EMPTY_JSON_INFORMATIVA)
    resultados_proceso_df = harmonize_dataframe(resultados_proceso_df, cols_to_keep)
    process_time = time() - start_time

    update_resultados_with_info_and_metadata(resultados_proceso_df, resultado_info_json, id_oficio, has_tables,
                                             oficio_texto, resultados_json, process_time, llm_model, oficio, num_tokens)
    return resultados_proceso_df


def process_batch(i, batch_md, client, llm_model, temperature, prompt_extraction_table):
    """
    Process each batch, including customization of messages and GPT extraction.
    """
    try:
        if ANONYMIZATION_SWITCH:
            batch_md, _ = anonymizer.anonymize_entities(batch_md)
        # Customize list with string
        messages_extract_tables = customize_list_with_string(batch_md, prompt_extraction_table)

        # Extract and parse to Markdown with retry logic
        resultados_md = extract_and_parse_to_md(messages_extract_tables, client, llm_model, temperature)

        # Convert markdown to DataFrame
        resultado_df_batch = markdown_to_dataframe(resultados_md)

        # Return DataFrame if valid
        if resultado_df_batch.shape[1] == NUM_COLS_TABLE:
            return i, resultado_df_batch
    except Exception as e:
        print(f"Problems with batch {i} of this oficio: {e}")
    return i, None

def process_oficio_with_tables(imagen_oficio,
                               oficio,
                               num_oficios,
                               textract_client,
                               client,
                               llm_model,
                               temperature,
                               prompt_extraction_table,
                               prompt_extraction_info_proceso,
                               prompt_model=None):
    """
    Processes a document image by extracting tables, converting them to DataFrames, batching,
    extracting additional information from text, and merging all extracted data.

    Parameters:
    - imagen_oficio: The image of the document to be processed.
    - oficio: The text content of the document.
    - textract_client: The client used to interact with AWS Textract for table extraction.
    - client: The API client for language llm_model completions.
    - llm_model (str): The language llm_model identifier.
    - temperature (float): The temperature setting for the llm_model's responses.
    - prompt_extraction_table: The prompt template for extracting table data.
    - prompt_extraction_info_proceso: The prompt template for extracting additional information from text.

    Returns:
    - pd.DataFrame: The DataFrame containing the merged and processed data from the document image.
    """

    # Extract tables from the document image

    if num_oficios==1:
        all_tables_md = extract_tables_from_images_to_md_parallel(imagen_oficio, textract_client)
        print("Splitting markdown table into batches")
        tables_md_batches = split_markdown_into_batches(all_tables_md,
                                                        max_batches=MAX_BATCHES_TABLES,
                                                        lines_per_batch=LINES_PER_BATCH)
        print(f"Processing {len(tables_md_batches)} batches")
        # Process each batch of table data
        resultado_df_batches = [None] * len(tables_md_batches)
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_GPT) as executor:
            # Submit tasks for parallel processing
            futures = [executor.submit(process_batch, i, batch_md, client, llm_model, temperature, prompt_extraction_table)
                    for i, batch_md in enumerate(tables_md_batches)]

            # Collect the results as tasks complete
            for future in as_completed(futures):
                i, result = future.result()
                if result is not None:
                    resultado_df_batches[i] = result  # Store result in the correct index
                else:
                    print(f"Batch {i} failed")
        # Filter out any None results (if there were any failures) and concatenate the valid ones
        resultado_df_batches = [df for df in resultado_df_batches if df is not None]

        if resultado_df_batches:
            all_batches_df = pd.concat(resultado_df_batches)
            print(all_batches_df.head())
    # Extract tables from the document image
    else:
        all_tables_md = extract_tables_from_images_to_md(imagen_oficio, textract_client)
        print("Splitting markdown table into batches")
        tables_md_batches = split_markdown_into_batches(all_tables_md,
                                                        max_batches=MAX_BATCHES_TABLES,
                                                        lines_per_batch=LINES_PER_BATCH)
        print(f"Processing {len(tables_md_batches)} batches")
        # Process each batch of table data
        resultado_df_batches = []
        for i, batch_md in enumerate(tables_md_batches):
            if ANONYMIZATION_SWITCH:
                batch_md, _ = anonymizer.anonymize_entities(batch_md)
                print(batch_md)
            messages_extract_tables = customize_list_with_string(batch_md, prompt_extraction_table)
            resultados_md = extract_and_parse_to_md(messages_extract_tables, client, llm_model, temperature)

            try:
                resultado_df_batch = markdown_to_dataframe(resultados_md)
                if resultado_df_batch.shape[1] == NUM_COLS_TABLE:
                    resultado_df_batches.append(resultado_df_batch)
            except:
                print(f"Problems with batch {i} of this oficio")
                pass

            all_batches_df = pd.concat(resultado_df_batches)
            print(all_batches_df.head())

    if all_batches_df.shape[0]==0:
        print("Table extraction not working, will use standard prompt")
        return None, oficio, {}, llm_model, 0

    # Extract and process specific information from oficio text
    print("Process the rest of the text")
    if len(oficio) > 1:
        oficio = "\n".join([oficio[0], oficio[-1]])
    else:
        oficio = "\n".join(oficio)
    # print("oficio: ", oficio)
    if ANONYMIZATION_SWITCH:
        oficio, _ = anonymizer.anonymize_entities(oficio)
        # print(oficio)
    message_extract_info_proceso = customize_list_with_string(oficio, prompt_extraction_info_proceso)
    # Check number of tokens:
    num_tokens = num_tokens_from_messages(message_extract_info_proceso, llm_model)
    # Get the results for the info
    resultados_json = extract_and_parse_to_json(message_extract_info_proceso, client, llm_model, temperature,
                                                prompt_model=prompt_model)
    resultados_proceso_df = pd.DataFrame([resultados_json])


    print("Concatenate both results, table and text")
    # Reset the index of both DataFrames to ensure they have unique indexes
    all_batches_df_reset = all_batches_df.reset_index(drop=True)
    resultados_proceso_df_reset = resultados_proceso_df.reset_index(drop=True)

    # Repeat the row of extracted information to match the number of rows in table data
    resultados_proceso_df_repeated = pd.concat([resultados_proceso_df_reset] * len(all_batches_df_reset),
                                               ignore_index=True)

    # Merge table data with extracted information
    result_df = pd.concat([all_batches_df_reset, resultados_proceso_df_repeated], axis=1)

    return result_df, oficio, resultados_json, llm_model, num_tokens


def process_oficio_without_tables(oficio,
                                  client,
                                  llm_model,
                                  temperature,
                                  prompt_extraction_standard,
                                  EMPTY_JSON_RETENCION_SUSPENSION_REMISION,
                                  prompt_model=None):
    """
    Processes an 'oficio' text to extract information without table data, handles retries,
    and returns a DataFrame with the flattened JSON results.

    Parameters:
    - oficio (str): The text content of the oficio to be processed.
    - client: The API client used for making requests to the language llm_model.
    - llm_model (str): The identifier for the language llm_model.
    - temperature (float): The temperature setting for the llm_model's responses.
    - prompt_extraction_standard (list): The template for customizing the extraction message.
    - EMPTY_JSON_RETENCION_SUSPENSION_REMISION (dict): A predefined JSON structure to use in case of extraction failure.

    Returns:
    - pd.DataFrame: A DataFrame containing the flattened JSON results.
    """
    # Customize the extraction message with the specific oficio

    oficio = "\n".join(oficio)
    # print("oficio: ", oficio)
    if ANONYMIZATION_SWITCH:
        oficio, _ = anonymizer.anonymize_entities(oficio)
        # print(oficio)
    messages_extract_standard = customize_list_with_string(oficio,
                                                           prompt_extraction_standard)

    num_tokens = num_tokens_from_messages(messages_extract_standard, llm_model)

    try:
        # Attempt to extract and parse information to JSON
        resultados_json = extract_and_parse_to_json(messages_extract_standard,
                                                    client,
                                                    llm_model,
                                                    temperature,
                                                    prompt_model=prompt_model
                                                    )
    except ValueError as e:
        # Handle failures by using a predefined empty result structure
        print("Intentos de llamar OpenAI excedio con error : ", e)
        resultados_json = EMPTY_JSON_RETENCION_SUSPENSION_REMISION

    # Flatten the extracted JSON to handle any embedded dictionaries
    resultados_json_flattened = flatten_dictionary(resultados_json)
    # Convert the flattened JSON to a DataFrame
    if resultados_json_flattened:
        resultados_proceso_df = pd.DataFrame(resultados_json_flattened)
    return resultados_proceso_df, oficio, resultados_json, llm_model, num_tokens


def update_resultados_with_info_and_metadata(resultados_proceso_df, resultado_info_json, id_oficio, has_tables,
                                             oficio_texto, resultados_json, process_time, llm_model, oficio,
                                             num_tokens):
    resultados_proceso_df["FECHA DE PUBLICACION DE LA CARTA CIRCULAR"] = resultado_info_json[
        "FECHA DE PUBLICACION DE LA CARTA CIRCULAR"]
    resultados_proceso_df["NUMERO DE CARTA CIRCULAR"] = resultado_info_json["NUMERO DE CARTA CIRCULAR"]
    resultados_proceso_df["IDENTIFICADOR UNICO QUE REGISTRA ASFI"] = id_oficio
    # Metadata columns
    resultados_proceso_df["TABLES"] = has_tables
    resultados_proceso_df["OFICIO_TEXTO"] = oficio_texto
    resultados_proceso_df["OUTPUT_JSON"] = str(resultados_json)
    resultados_proceso_df["PROCESSING_TIME"] = process_time
    resultados_proceso_df["LLM_MODEL"] = llm_model
    resultados_proceso_df["NUM_PAGES"] = len(oficio)
    resultados_proceso_df["NUM_TOKENS"] = num_tokens
    print("Added columns info extraction and metadata")


def fill_carta_template(resultados_proceso_df):
    cartas_informativas = []
    formatted_date = get_todays_date_in_spanish()

    for i, row in resultados_proceso_df.iterrows():
        carta = f"""{row["NOMBRE DE LA CIUDAD DEL SOLICITANTE"]}, {formatted_date}
    COS/REQ/XXXX/2024

    Señora
    {row["NOMBRE DE LA AUTORIDAD SOLICITANTE"]}
    {row["NOMBRE DEL JUZGADO"]}
    DE LA CIUDAD DE {row["NOMBRE DE LA CIUDAD DEL SOLICITANTE"]} 
    Presente. –

    De nuestra consideración:

    Dando cumplimiento a la Carta Circular {row["NUMERO DE CARTA CIRCULAR"]} emitida por la Autoridad de Supervisión del Sistema Financiero (ASFI), de fecha {row["FECHA DE PUBLICACION DE LA CARTA CIRCULAR"]}, 
    dentro del proceso {row["TIPO DE PROCESO"]} seguido por {row["DEMANDANTES"]} en contra de {row["DEMANDADOS"]}, 
    caso {row["TIPO DE CASO"]}: {row["NUMERO DOCUMENTO DE RESPALDO"]}, su oficio de fecha {row["FECHA DE PUBLICACION"]}, informamos lo siguiente: 

    - 	{row["INSTRUCCION ESPECIFICA PARA EL BANCO"]}

    Con este motivo, saludamos a usted con nuestras consideraciones más distinguidas.

    Atentamente, 
    Banco XXX
    """
        cartas_informativas.append(carta)
    return cartas_informativas


def classify_oficios(oficios, lista_oficios_ids, client, llm_model, temperature):
    """
    Clasifica una lista de oficios y determina su categoría.

    Args:
        oficios (list of list of str): Una lista de oficios, cada uno representado como una lista de cadenas de texto.
        lista_oficios_ids (list of str): Una lista de identificadores únicos para cada oficio.
        client (object): Cliente para interactuar con el modelo de lenguaje.
        llm_model (str): Nombre del modelo de lenguaje a utilizar.
        temperature (float): Valor de la temperatura para el modelo de lenguaje, que controla la aleatoriedad de las respuestas.

    Returns:
        dict: Un diccionario donde las claves son los identificadores de los oficios y los valores son las categorías
              ("RETENCION", "SUSPENSION", "REMISION", "INFORMATIVA") o None si no se pudo clasificar.
    """
    oficio_clasificacion = {}
    for oficio_texto_lista, oficio_id in zip(oficios, lista_oficios_ids):
        oficio_texto = " ".join(oficio_texto_lista)
        if ANONYMIZATION_SWITCH:
            oficio_texto, _ = anonymizer.anonymize_entities(oficio_texto)
        messages_classification = customize_list_with_string(oficio_texto, prompt_clasificar_oficio)
        # Count number of tokens:
        num_tokens = num_tokens_from_messages(messages_classification, llm_model)
        # Si el numero de tokens es mas que 120k entonces solo tomamos la primera pagina
        if num_tokens > MAX_TOKENS_GPT_4o:
            print("Arreglar el hecho de que el oficio puede tener demasiados tokens")
            oficio_primera_pagina = oficio_texto_lista[0]
            messages_classification = customize_list_with_string(oficio_primera_pagina, prompt_clasificar_oficio)
        resultados = extract_and_parse_to_text(messages_classification,
                                               client,
                                               llm_model,
                                               temperature)
        if resultados in ["RETENCION", "SUSPENSION", "REMISION", "INFORMATIVA"]:
            oficio_clasificacion[oficio_id] = resultados
        else:
            oficio_clasificacion[oficio_id] = None
    return oficio_clasificacion


def extract_info_normativa(oficio_texto, client, llm_model, temperature):
    """
    Extrae información normativa de un texto de oficio y la devuelve en un DataFrame armonizado.
    Args:
        oficio_texto (str): El texto del oficio del cual se extraerá la información normativa.
        client (object): Cliente para interactuar con el modelo de lenguaje.
        llm_model (str): Nombre del modelo de lenguaje a utilizar.
        temperature (float): Valor de la temperatura para el modelo de lenguaje, que controla la aleatoriedad de las respuestas.
    Returns: pandas.DataFrame: DataFrame con la información normativa extraída y armonizada.
    """
    if ANONYMIZATION_SWITCH:
        oficio_texto = anonymizer.anonymize_entities(oficio_texto)
    messages_extraction_normativa = customize_list_with_string(oficio_texto, prompt_extraction_normativa)
    num_tokens = num_tokens_from_messages(messages_extraction_normativa, llm_model)
    resultados_json = extract_and_parse_to_json(messages_extraction_normativa,
                                                client,
                                                llm_model,
                                                temperature)

    resultados_df = pd.DataFrame([resultados_json])
    resultados_df = harmonize_dataframe(resultados_df, COLS_NORMATIVA)

    return resultados_df


# Processing Oficios:
def process_oficios_in_parallel(oficios,
                                oficio_clasificacion,
                                imagenes_oficios_separados,
                                ids_oficios,
                                resultado_info_json,
                                client,
                                llm_model,
                                temperature,
                                max_workers=MAX_WORKERS,
                                ):
    """
    Procesa oficios en paralelo y concatena los resultados en un DataFrame final.
    Args:
        oficios (list of list of str): Lista de oficios, cada uno representado como una lista de cadenas de texto.
        oficio_clasificacion (dict): Diccionario con la clasificación de cada oficio.
        imagenes_oficios_separados (list of str): Lista de rutas de imágenes correspondientes a cada oficio.
        ids_oficios (list of str): Lista de identificadores únicos para cada oficio.
        resultado_info_json (dict): Información JSON adicional necesaria para el procesamiento.
        client (object): Cliente para interactuar con el modelo de lenguaje.
        llm_model (str): Nombre del modelo de lenguaje a utilizar.
        temperature (float): Valor de la temperatura para el modelo de lenguaje, que controla la aleatoriedad de las respuestas.
        max_workers (int, optional): Número máximo de hilos a utilizar en el procesamiento paralelo. Por defecto es 5.
    Returns: pandas.DataFrame: DataFrame final que contiene los resultados concatenados y ordenados.
    """

    def process_oficio(oficio, imagen_oficio, id_oficio):
        """
        Procesa un oficio individual y devuelve el resultado como un DataFrame.
        Args:
            oficio (list of str): El texto del oficio.
            imagen_oficio (str): La ruta de la imagen del oficio.
            id_oficio (str): El identificador único del oficio.
        Returns: pandas.DataFrame: DataFrame con el resultado del procesamiento del oficio.
        """
        # Adapted to process individual items directly
        resultados = process_each_oficio(resultado_info_json,
                                           oficio,
                                           oficio_clasificacion,
                                           imagen_oficio,
                                           id_oficio,
                                           len(ids_oficios),
                                           client,
                                           llm_model,
                                           temperature)
        return resultados
    results = parallel_process(process_oficio, oficios, imagenes_oficios_separados, ids_oficios,
                            max_workers=MAX_WORKERS)
    # Concatenate all result DataFrames into a single DataFrame and sort
    print(f"Concatenating {len(results)} results resetting the index for secondary sorting")
    # Reset the index of each DataFrame in 'results' and keep the original index as a column
    for j,df in enumerate(results):
        if df is None:
            print(f"For oficio with index {j} the result is None and should be checked")
    results_with_index = [df.reset_index() for df in results if df is not None]
    # Concatenate results with the original index preserved
    final_result_df = pd.concat(results_with_index, ignore_index=True)

    print("Ordering by ASFI ID")
    if len(ids_oficios) > 1:
        final_result_df['sort_order'] = final_result_df['IDENTIFICADOR UNICO QUE REGISTRA ASFI'].apply(
            lambda x: ids_oficios.index(x) if x in ids_oficios else len(ids_oficios)
        )
        # Sort by 'sort_order' and then by the original index to keep the initial order as a secondary criterion
        final_result_df.sort_values(by=['sort_order', 'index'], inplace=True)
        # Drop the 'sort_order' column as it's no longer needed after sorting
        final_result_df.drop(['sort_order', 'index'], axis=1, inplace=True)
        # Optionally, if you want to remove the 'index' column as well, you can drop it or set it as the index
        final_result_df.reset_index(drop=True, inplace=True)  # If you want to drop the 'index' column

    return final_result_df


def process_each_oficio(resultado_info_json,
                        oficio,
                        oficio_clasificacion,
                        imagen_oficio,
                        id_oficio,
                        num_oficios,
                        client,
                        llm_model,
                        temperature=0):
    """
    Extracts information if each oficio is on a single page:
    """
    print(f"Processing Oficio with ID {id_oficio}")
    print(f"Check if it has tables")

    tipo_oficio = oficio_clasificacion[id_oficio]

    if tipo_oficio in ["RETENCION", "SUSPENSION", "REMISION"]:
        resultado_df = extraer_informacion_de_cada_retencion_suspension_remision(resultado_info_json,
                                                                                 oficio,
                                                                                 imagen_oficio,
                                                                                 id_oficio,
                                                                                 num_oficios,
                                                                                 client,
                                                                                 llm_model,
                                                                                 temperature,
                                                                                 prompt_extraction_standard,
                                                                                 prompt_extraction_table,
                                                                                 prompt_extraction_tabla_info_proceso,
                                                                                 COLS_RETENCION_SUSPENSION_REMISION_NO_CI_ABREVIACION,
                                                                                 prompt_model_standard=CIRCULAR_STANDARD_MODEL,
                                                                                 prompt_model_info_proceso=CIRCULAR_TABLA_INFO_PROCESO_MODEL
                                                                                 )

    elif tipo_oficio == "INFORMATIVA":
        resultado_df = extraer_informacion_de_cada_informativa(resultado_info_json,
                                                               oficio,
                                                               id_oficio,
                                                               client,
                                                               llm_model,
                                                               temperature,
                                                               prompt_informativas,
                                                               COLS_INFORMATIVA)
    return resultado_df
