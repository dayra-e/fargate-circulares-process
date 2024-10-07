import io
import locale
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
from time import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


def load_tiff_pages_from_image(img):
    """
    Load all pages from a multi-page TIFF file and return them as a list of images.

    :param img: image openend with PIL
    :return: List of PIL Image objects, each representing a page of the TIFF file.
    """
    images = []
    # with Image.open(tiff_path) as img:
    for frame_num in range(img.n_frames):
        img.seek(frame_num)
        images.append(img.copy())  # Copy the current frame and add it to the list

    return images


def image_to_bytes(img, format="PNG"):
    """
    Convert an image into bytes with the given format
    """
    buffer = io.BytesIO()
    img.save(buffer, format=format)  # Saving image to buffer
    image_bytes = buffer.getvalue()
    return image_bytes


def is_dict_with_lists(d):
    """
    Determines whether any value in a dictionary is a list that contains only non-dictionary elements.
    Args:
        d (dict): The dictionary to check.
    Returns:
        bool: True if there is at least one such list, False otherwise.
    """
    # Iterate over dictionary values
    for value in d.values():
        # Check if value is a list and does not contain dictionaries
        if isinstance(value, list) and not any(isinstance(item, dict) for item in value):
            return True  # Return True if at least one list without dictionaries is found
    return False  # Return False if no such list is found


def is_dict_with_list_of_dicts(d):
    """
    Checks if any value in a dictionary is a list composed entirely of dictionaries.
    Args: d (dict): The dictionary to examine.
    Returns: bool: True if at least one value is a list containing only dictionaries; False otherwise.
    """
    # Iterate over dictionary values
    for value in d.values():
        # Check if value is a list containing only dictionaries
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return True  # Return True if at least one list with dictionaries is found
    return False  # Return False if no such list is found


def flatten_dictionary(d):
    """
    Flattens a dictionary that may contain nested lists of dictionaries or other elements, ensuring all
    embedded lists are handled according to their contents.
    Args: d (dict): The dictionary to flatten, which may contain lists of non-dictionaries, lists of dictionaries, or direct key-value pairs.
    Returns: list of dict: A list of dictionaries where each dictionary represents a flattened version of the original dictionary.
    Raises: ValueError: If lists of non-dictionary items have varying lengths.
    Description:
        The function checks for and handles three cases:
        1. Direct key-value pairs where the value is not a list.
        2. Lists of non-dictionaries ensuring they are of uniform length.
        3. Lists of dictionaries, merging each dictionary with the direct key-value pairs.
        It constructs a list where each element is a dictionary representing a "flattened" version of the input.
    """
    # Initialize an empty list to hold the flattened dictionaries
    flattened_dicts = []
    # Extract the key-value pairs that are not lists (direct key-value pairs)
    direct_pairs = {k: v for k, v in d.items() if not isinstance(v, list)}
    if is_dict_with_lists(d):
        # Separate keys with list values from direct key-value pairs
        list_pairs = {k: v for k, v in d.items() if isinstance(v, list)}
        # Check if any of the list is empty and handle accordingly
        if all(len(v) == 0 for v in list_pairs.values()):
            # If all lists are empty, treat it as a direct_pairs case
            flattened_dicts.append(direct_pairs)
        else:
            # Ensure all non-empty lists are of the same length
            list_lengths = [len(v) for v in list_pairs.values() if len(v) > 0]
            if len(set(list_lengths)) != 1:
                raise ValueError("All non-empty lists must be of the same length")
            # Use zip to iterate over list values simultaneously, ignoring empty lists
            for zipped_values in zip(*[v for v in list_pairs.values() if len(v) > 0]):
                # Create a dictionary that merges direct_pairs with the zipped values
                # Map zipped values back to their corresponding keys
                keys_for_zipping = [k for k, v in list_pairs.items() if len(v) > 0]
                merged_dict = {**direct_pairs, **dict(zip(keys_for_zipping, zipped_values))}
                flattened_dicts.append(merged_dict)
    elif is_dict_with_list_of_dicts(d):
        # Handle lists of dictionaries
        for key, value in d.items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                for item in value:
                    merged_dict = {**direct_pairs, **item}
                    flattened_dicts.append(merged_dict)
    else:
        # Case where there are no lists, or lists are empty, directly return the dictionary in a list
        return [d]
    return flattened_dicts


def harmonize_dataframe(df, cols_to_keep):
    """
    Harmonizes a DataFrame to ensure it contains only the specified columns.

    Parameters:
    - df (DataFrame): The original DataFrame to be harmonized.
    - cols_to_keep (list): A list of column names to keep in the DataFrame.

    Returns:
    - DataFrame: A new DataFrame containing only the columns specified in cols_to_keep.
                 Columns not present in the original DataFrame are filled with NaNs.
    """
    # Create a dictionary where keys are the columns to keep and values are NaN
    # This ensures that if a column is missing, it will be created with NaN values
    missing_cols = {col: [None] * len(df) for col in cols_to_keep if col not in df.columns}

    # Add the missing columns to the DataFrame
    df = df.assign(**missing_cols)

    # Reorder and select only the columns to keep, filling missing ones with NaN
    return df[cols_to_keep]


def get_todays_date_in_spanish():
    """
    Returns today's date in a literal Spanish format.

    :return: A string representing today's date in the format "3 de mayo de 2024".
    """
    # Attempt to set the locale to Spanish
    try:
        locale.setlocale(locale.LC_TIME, 'es_ES')  # Try for Unix/Linux
    except locale.Error:
        try:
            locale.setlocale(locale.LC_TIME, 'Spanish_Spain')  # Try for Windows
        except locale.Error:
            return "Locale not supported"

    # Get today's date
    today = datetime.now()

    # Format the date in the literal Spanish format
    formatted_date = today.strftime('%-d de %B de %Y')

    return formatted_date


def create_metadata_instances(resultados_metadata, request, image_instance):
    """
    Create metadata instances from dataframe rows.

    Args:
    resultados_metadata (DataFrame): DataFrame containing the metadata.
    request (Request): The request object containing user information.
    image_instance (Image): The image instance associated with the metadata.

    Returns:
    List[CircularMetadata]: A list of CircularMetadata instances.
    """
    all_metadata_instances = []
    for index, row in resultados_metadata.iterrows():
        metadata_instance = CircularMetadata(
            user=request.user,
            fecha_publicacion=parse_custom_date_with_reformatting(row['FECHA DE PUBLICACION DE LA CARTA CIRCULAR']),
            numero_circular=row['NUMERO DE CARTA CIRCULAR'],
            identificador_asfi=row['IDENTIFICADOR UNICO QUE REGISTRA ASFI'],
            tables=bool(row['TABLES']),  # Ensure the value is interpreted as a boolean
            oficio_texto=row['OFICIO_TEXTO'],
            image=image_instance,
            caso=row["CASO"],
            oficio_inicios=row["OFICIO_INICIOS"],
            output_json=row["OUTPUT_JSON"],
            processing_time=row["PROCESSING_TIME"],
            llm_model=row["LLM_MODEL"],
            num_pages=row["NUM_PAGES"],
            num_tokens=row["NUM_TOKENS"]
        )
        all_metadata_instances.append(metadata_instance)

    return all_metadata_instances


def image_to_base64(image):
    """
    Convert an image to a base64 encoded string.

    Parameters:
    - image: The image object to convert.

    Returns:
    - image_base64: (str)The base64 encoded string representation of the image.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64


def meausure_execution_time(title=''):
    def medir_tiempo_wa(func):
        def wrapper(*args, **kwargs):
            inicio = time()
            resultado = func(*args, **kwargs)
            tiempo_ejecucion = time() - inicio
            print(f"TIME:: Time: {tiempo_ejecucion:.4f} seg | {func.__name__} | {title}")
            return resultado

        return wrapper

    return medir_tiempo_wa
# Tables utils
def calculate_area(bbox):
     # Acceder al bbox de cada objeto ExtractedTable
    x_min = bbox.x1
    y_min = bbox.y1
    x_max = bbox.x2
    y_max = bbox.y2
    # Calcular el ancho y el alto
    ancho = x_max - x_min
    alto = y_max - y_min
    # Calcular el área
    area = ancho * alto
    return area

def parallel_process(func, *args, max_workers=None, pool_type='thread'):
    results = [None] * len(args[0])

    def task(idx, *_arg):
        print(f"POOL:: {pool_type}: Running task id: {idx}")
        return idx, func(*_arg)

    if pool_type not in ('thread', 'proc'):
        raise Exception("PARALLEL:: pool type must be thread or proc")
    if pool_type == 'thread':
        pool = ThreadPoolExecutor
    else:
        # FIXME Exception = results_with_index = [df.reset_index() for df in results]
        raise NotImplementedError("Check some errors")
        pool = ProcessPoolExecutor

    with pool(max_workers=max_workers) as executor:
        futures = [executor.submit(task, idx, *_arg) for idx, _arg in enumerate(zip(*args))]
        for future in as_completed(futures):
            try:
                _idx, _res = future.result()
                results[_idx] = _res
            except Exception as exc:
                print(f'Generated an exception: {exc}')
    return results


if __name__ == '__main__':
    @meausure_execution_time()
    def prueba1(a, b):
        return a + b


    def prueba2(a: list, b: list):
        parallel_process(prueba1, a, b)


    print(prueba1(1, 4))

    print(prueba2([1, 2, 3], [4, 5, 6]))
