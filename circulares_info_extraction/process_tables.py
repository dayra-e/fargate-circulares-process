try:
    from img2table.document.image import Image as ImageTable
except:
    try:
        from img2table.document.image import Image as ImageTable    
    except Exception as e:
        print(f"Error: {e}")
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from circulares_info_extraction.utils_etl import (image_to_bytes, 
                                                      calculate_area)
from circulares_info_extraction.config import LoadConfig
import random

config = LoadConfig()
config.set_section('tables')
THRESHOLD_TABLE = config.parameter("threshold")
MIN_CONFIDENCE_TABLE = config.parameter("min_confidence_table")
MIN_AREA = config.parameter("min_area")
HEADER_POSITION =  config.parameter("header_position")
FOOTER_POSITION = config.parameter("footer_position")
SAMPLE_PERCENTAGE = config.parameter("sample_percentage")
MAX_PAGES = config.parameter("max_pages")

config.set_section('concurrency')
MAX_WORKERS = config.parameter("max_workers_textract")

def check_tables_one_page(img, min_confidence=MIN_CONFIDENCE_TABLE):
    """
    Checks if there are tables in oficios that have one page.
    We use a heuristic that if there is only one table and it is small (less or equal to 2x2)
    then we don't consider it has tables
    """
    image_bytes = image_to_bytes(img)
    # Start ImageTable
    doc = ImageTable(image_bytes)
    # Table extraction
    extracted_tables = doc.extract_tables(min_confidence=min_confidence)
    # If there is only no tables
    if len(extracted_tables) == 0:
        return False
    # if there is one table and it's size is less or equal to 2x2
    elif len(extracted_tables) == 1:
        df = extracted_tables[0].df
        if (df.shape[0] <= 2 and df.shape[1] <= 3) or (df.shape[0] <= 10 and df.shape[1] <= 1):
            return False
        else:
            return True
    # If there is more than one table:
    return True


def check_tables_simple(imagenes_oficios):
    """
    Check if any of the pages contains a table
    """
    for img in imagenes_oficios:
        image_bytes = image_to_bytes(img)
        # Start ImageTable
        doc = ImageTable(image_bytes)
        # Table extraction
        extracted_tables = doc.extract_tables(min_confidence=MIN_CONFIDENCE_TABLE)

        if len(extracted_tables) > 0:
            return True
    return False
def check_if_table_is_valid(bbox, height):
    """
    Check if a table is valid based on area and position
    param table: list of img2table.table.Table
    """
    y_min = bbox.y1
    y_max = bbox.y2
    if calculate_area(bbox) < MIN_AREA:
        return False
    elif y_min < HEADER_POSITION and y_max < HEADER_POSITION:
        return False
    elif y_min >(height-FOOTER_POSITION)  and y_max > (height-FOOTER_POSITION):        
        return False
    else:
        return True
    


def check_tables(imagenes_oficios, threshold=THRESHOLD_TABLE, min_confidence=MIN_CONFIDENCE_TABLE):
    """
    Check if more than or equal to a specified percentage of the pages contain a table.
    Exits early if the specified threshold is met or cannot be met with the remaining pages.

    :param imagenes_oficios: List of images to check for tables.
    :param threshold: The minimum fraction of pages that need to contain tables for the function to return True.
    :param min_confidence: The minimum confidence level required to consider a table extraction successful.
    :return: True if the percentage of pages with tables is greater than or equal to the threshold, False otherwise.
    """
    total_pages = len(imagenes_oficios)
    pages_with_tables = 0

    if total_pages == 1:
        return check_tables_one_page(imagenes_oficios[0])
    elif total_pages <= MAX_PAGES:
        for index, img in enumerate(imagenes_oficios):
            image_bytes = image_to_bytes(img)
            # Start ImageTable
            doc = ImageTable(image_bytes)
            # Table extraction
            extracted_tables = doc.extract_tables(min_confidence=min_confidence)
            if extracted_tables:
                for table in extracted_tables:
                    _, height = img.size
                    if check_if_table_is_valid(table.bbox, height): 
                      print("table is valid")
                      pages_with_tables += 1
                      break
            # Calculate the maximum possible tables percentage for the remaining pages
            max_possible_tables_percentage = (pages_with_tables + (total_pages - (index + 1))) / total_pages

            # Check if it's already impossible to meet or exceed the threshold with the remaining pages
            if pages_with_tables / total_pages >= threshold:
                return True
            elif max_possible_tables_percentage < threshold:
                print("max_possible", max_possible_tables_percentage)
                return False
        return False
    else:
        print(f"Extracted tables >100: {extracted_tables}")
        sample_size = int(len(imagenes_oficios)*SAMPLE_PERCENTAGE)
        # Tomar una muestra de acuerdo al tamaño de la lista
        sample = random.sample(imagenes_oficios, sample_size)
        for index, img in enumerate(sample):
            image_bytes = image_to_bytes(img)
            # Start ImageTable
            doc = ImageTable(image_bytes)
            # Table extraction
            extracted_tables = doc.extract_tables(min_confidence=min_confidence)

            if extracted_tables:
                for table in extracted_tables:
                    _, height = img.size
                    if check_if_table_is_valid(table.bbox, height): 
                      pages_with_tables += 1
                      break
            # Calculate the maximum possible tables percentage for the remaining pages
            max_possible_tables_percentage = (pages_with_tables + (sample_size - (index + 1))) / sample_size

            # Check if it's already impossible to meet or exceed the threshold with the remaining pages
            if pages_with_tables / sample_size >= threshold:
                print(extracted_tables)
                return True
            elif max_possible_tables_percentage < threshold:
                return False
        return False

def extract_tables_to_dataframes(response):
    """
    Extracts tables from a AWS Textract response and converts them into pandas DataFrames.
    Input:response: dict
        The response dictionary from AWS Textract after processing a document.
        It contains details of detected 'Blocks' like tables, cells, words, etc.
    Output:dataframes: list of pd.DataFrame
        A list containing a pandas DataFrame for each table found in the document.
        Each DataFrame represents a table, where the first row of the DataFrame is
        assumed to be the column headers.
    """

    def map_blocks(blocks, block_type):
        """
        Maps each block of a specified type to its ID.
        Input:
            blocks: list of dicts
                The 'Blocks' part of the Textract response.
            block_type: str
                The type of block to filter for (e.g., 'TABLE', 'CELL', 'WORD').
        Output:
            A dictionary mapping block IDs to their full block information for all blocks of the specified type.
        """
        return {
            block['Id']: block
            for block in blocks
            if block['BlockType'] == block_type
        }

    def get_children_ids(block):
        """
        Yields the IDs of child blocks for a given block.
        Input:block: dict
        A block dictionary from the Textract response.

        Output: Yields the IDs of the block's children, if any.
        """
        for rels in block.get('Relationships', []):
            if rels['Type'] == 'CHILD':
                yield from rels['Ids']

    blocks = response['Blocks']
    tables = map_blocks(blocks, 'TABLE')
    cells = map_blocks(blocks, 'CELL')
    words = map_blocks(blocks, 'WORD')
    selections = map_blocks(blocks, 'SELECTION_ELEMENT')

    dataframes = []

    for table in tables.values():
        # Determine all the cells that belong to this table
        table_cells = [cells[cell_id] for cell_id in get_children_ids(table)]

        # Determine the table's number of rows and columns
        n_rows = max(cell['RowIndex'] for cell in table_cells)
        n_cols = max(cell['ColumnIndex'] for cell in table_cells)
        content = [[None for _ in range(n_cols)] for _ in range(n_rows)]

        # Fill in each cell
        for cell in table_cells:
            cell_contents = [
                words[child_id]['Text'] if child_id in words else selections[child_id]['SelectionStatus']
                for child_id in get_children_ids(cell)
            ]
            i = cell['RowIndex'] - 1
            j = cell['ColumnIndex'] - 1
            content[i][j] = ' '.join(cell_contents)

        # We assume that the first row corresponds to the column names
        dataframe = pd.DataFrame(content[1:], columns=content[0])
        dataframes.append(dataframe)

    return dataframes


def extract_tables_to_markdown(response):
    """
    Extracts tables from an AWS Textract response and converts them into markdown tables.
    Input:
        response: dict
            The response dictionary from AWS Textract after processing a document.
            It contains details of detected 'Blocks' like tables, cells, words, etc.
    Output:
        markdown_tables: list of str
            A list containing a markdown table for each table found in the document.
    """

    def map_blocks(blocks, block_type):
        return {
            block['Id']: block
            for block in blocks
            if block['BlockType'] == block_type
        }

    def get_children_ids(block):
        for rels in block.get('Relationships', []):
            if rels['Type'] == 'CHILD':
                yield from rels['Ids']

    blocks = response['Blocks']
    tables = map_blocks(blocks, 'TABLE')
    cells = map_blocks(blocks, 'CELL')
    words = map_blocks(blocks, 'WORD')
    selections = map_blocks(blocks, 'SELECTION_ELEMENT')

    markdown_tables = []

    for table in tables.values():
        table_cells = [cells[cell_id] for cell_id in get_children_ids(table)]
        n_rows = max(cell['RowIndex'] for cell in table_cells)
        n_cols = max(cell['ColumnIndex'] for cell in table_cells)
        content = [["" for _ in range(n_cols)] for _ in range(n_rows)]

        for cell in table_cells:
            cell_contents = [
                words[child_id]['Text'] if child_id in words else selections[child_id]['SelectionStatus']
                for child_id in get_children_ids(cell)
            ]
            i = cell['RowIndex'] - 1
            j = cell['ColumnIndex'] - 1
            content[i][j] = ' '.join(cell_contents).strip()

        # Convert to markdown table string
        markdown_table = "| " + " | ".join(content[0]) + " |\n"
        markdown_table += "|---" * n_cols + "|\n"  # Header separator
        for row in content[1:]:
            markdown_table += "| " + " | ".join(row) + " |\n"

        markdown_tables.append(markdown_table)

    return markdown_tables


def extract_tables_from_images_to_df(images, textract_client):
    """Extract tables from a list of PIL images."""
    tables = []
    print("Extracting tables from pages")
    for j, image in enumerate(images):
        print(f"Text extraction for page : {j}")
        # Convert the PIL Image to bytes
        image_bytes = image_to_bytes(image)
        # Call Amazon Textract to analyze the document for tables
        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['TABLES']
        )

        # Process the response to extract tables
        # The exact processing depends on your needs. For simplicity, here we just append the raw response.
        table = extract_tables_to_dataframes(response)
        tables.append(table)

    return tables

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2) + wait_random(0, 1))
def process_table_from_image(image, textract_client, index=None):
    """
    Process a single image to extract tables using Textract, with retry logic.
    """
    try:
        print(f"Table extraction with Textract for page {index}")
        # Convert the PIL Image to bytes
        image_bytes = image_to_bytes(image)
        # Call Amazon Textract to analyze the document for tables
        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['TABLES']
        )
        # Process the response to extract tables
        tables_md = extract_tables_to_markdown(response)
        return index, tables_md

    except Exception as e:
        print(f"Error processing table for page {index}: {e}")
        # Raise the exception to trigger the retry logic
        raise

def extract_tables_from_images_to_md_parallel(images, textract_client):
    """
    Extract tables from a list of PIL images in parallel using ThreadPoolExecutor, with retry logic.
    """
    tables_md = [[""] for _ in range(len(images))] # Placeholder for results

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Schedule the process_table_from_image function to be executed for each image
        futures = [executor.submit(process_table_from_image, image, textract_client, index)
                   for index, image in enumerate(images)]

        # Wait for all futures to complete and store results in their respective index
        for future in as_completed(futures):
            index, table_md = future.result()
            tables_md[index] = table_md
    
    tables_md_strs= [string for sublist in tables_md for string in sublist]
    all_table_md = "".join(tables_md_strs)
    return all_table_md

def extract_tables_from_images_to_md(images, textract_client):
    """Extract tables from a list of PIL images."""
    tables_md = []
    print("Extracting tables from pages")
    for j, image in enumerate(images):
        print(f"Table extraction with Textract for page : {j}")
        # Convert the PIL Image to bytes
        image_bytes = image_to_bytes(image)
        # Call Amazon Textract to analyze the document for tables
        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['TABLES']
        )

        # Process the response to extract tables
        # The exact processing depends on your needs. For simplicity, here we just append the raw response.
        tables_md += extract_tables_to_markdown(response)
    all_table_md = "".join(tables_md)
    return all_table_md


def flatten_list(input_list):
    flattened_list = []
    for item in input_list:
        if isinstance(item, list):
            # If the item is a list, extend the flattened list with the result of the recursive call
            flattened_list.extend(flatten_list(item))
        elif isinstance(item, pd.DataFrame):
            # If the item is a DataFrame, append it to the flattened list
            flattened_list.append(item)
        else:
            # Optionally handle other types or raise an error
            raise ValueError("List contains an item that is neither a DataFrame nor a list.")
    return flattened_list


def standarize_and_concat_tables(all_tables):
    """
    Standardizes column headers of a list of DataFrames to match the first DataFrame's headers and concatenates them into a single DataFrame.

    Parameters:
    - all_tables (list of pd.DataFrame): List of DataFrames with the first DataFrame serving as the header template.

    Returns:
    - pd.DataFrame: A single DataFrame with unified column names from all input DataFrames.

    Note:
    - Assumes all DataFrames are compatible in terms of column count and order after the first.
    """
    standard_columns = all_tables[0].columns

    # Initialize a list to hold all DataFrames with standardized columns
    standardized_dataframes = []

    # Process each DataFrame
    for i, df in enumerate(all_tables):
        if i == 0:
            # The first DataFrame is added directly
            standardized_dataframes.append(df)
        else:
            # Subsequent DataFrames are added with columns aligned to the first DataFrame
            # Ignoring their headers and using the column names from the first DataFrame
            df.columns = standard_columns  # Assign the standard column names to this DataFrame
            standardized_dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    concatenated_df = pd.concat(standardized_dataframes, ignore_index=True)
    return concatenated_df


def markdown_to_dataframe(markdown_string):
    """
    Converts a Markdown table string to a pandas DataFrame.

    Parameters:
    - markdown_string (str): A string containing a Markdown formatted table.

    Returns:
    - pd.DataFrame: A DataFrame containing the data from the Markdown table.
    """
    # Split the Markdown string by lines and filter out empty lines
    lines = [line.strip() for line in markdown_string.strip().split('\n') if line.strip()]

    # Extract headers
    headers = lines[0].split('|')[1:-1]  # Exclude the first and last empty strings
    headers = [header.strip() for header in headers]

    # Extract rows
    rows = [line.split('|')[1:-1] for line in lines[2:]]  # Skip the delimiter row
    rows = [[cell.strip() for cell in row] for row in rows]

    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)

    print(f"Resulting dataframe shape has {df.shape[0]} rows and {df.shape[1]} columns")

    return df


def split_markdown_into_batches(markdown_text, max_batches=10, lines_per_batch=200):
    """
    Splits a markdown table into specified parts, each with up to a certain number of lines including the header.
    Parameters:
    markdown_text (str): The entire markdown table as a string.
    max_batches (int): Maximum number of batches to split into.
    lines_per_batch (int): Maximum number of lines per batch, including the header.
    Returns:
    list of str: A list of markdown table strings.
    """
    # Split the markdown text into lines
    lines = markdown_text.strip().split('\n')

    # Ensure there's more than just a header; if not, just return the original text in a list
    if len(lines) <= lines_per_batch:
        return [markdown_text]

    # Extract the header and the separator
    header = lines[:2]

    # Prepare to accumulate batches
    batches = []
    current_batch = header.copy()  # Start the first batch with the header

    for line in lines[2:]:  # Skip the header
        current_batch.append(line)
        # If we reached the maximum lines per batch or foresee exceeding max_batches with current distribution
        if len(current_batch) == lines_per_batch or \
                (len(batches) == max_batches - 1 and len(lines) > len(current_batch) + lines_per_batch * (
                        max_batches - len(batches))):
            batches.append('\n'.join(current_batch))
            current_batch = header.copy()  # Reset current batch

    # Don't forget to add the last batch if it has any content beyond the header
    if len(current_batch) > 2:
        batches.append('\n'.join(current_batch))

    return batches


def batch_dataframe(df, batch_size=50):
    """
    Separates a DataFrame into a list of DataFrames, each with up to 'batch_size' rows.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be batched.
    - batch_size (int): The number of rows in each batch.

    Returns:
    - list of pd.DataFrame: A list containing subsets of the original DataFrame,
      each with up to 'batch_size' rows.
    """
    batches = []  # Initialize the list to hold batches
    num_rows = df.shape[0]

    for start_row in range(0, num_rows, batch_size):
        batches.append(df.iloc[start_row:start_row + batch_size])

    return batches
