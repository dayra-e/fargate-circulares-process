import pandas as pd

pd.set_option('display.max_columns', None)
from dotenv import load_dotenv

load_dotenv()
import sys

sys.path.append("../src/")
sys.path.append("../../account/")
import openai
import os

openai_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_key
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from api.circulares_info_extraction.utils_etl import (image_to_bytes,
                                                      image_to_base64)
from api.circulares_info_extraction.api_clients import (rekognition,
                                                        lambda_client)  # AWS Rekognition:
from api.circulares_info_extraction.config import LoadConfig

config = LoadConfig()

# For stamp recognition
config.set_section('stamp_recognition')
# For rekognition stamp recognition
AWS_REKOGNITION_MODEL_ID = config.parameter("aws_rekognition_model_id")
INITIAL_MIN_CONFIDENCE = config.parameter('initial_min_confidence')
CONFIDENCE_STEP_DOWN = config.parameter('confidence_step_down')
CONFIDENCE_STEP_UP = config.parameter('confidence_step_up')
MAX_ITERATIONS_STAMP_FINDING = config.parameter('max_iterations')
# For lambda stamp recognition
MIN_CONFIDENCE_LAMBDA = config.parameter('min_confindence_lambda')
MAX_WORKERS = config.parameter('max_workers')


def adjusted_find_stamp_confidence(imgs,
                                   target_stamps,
                                   aws_rekognition_model_id=AWS_REKOGNITION_MODEL_ID,
                                   initial_min_confidence=INITIAL_MIN_CONFIDENCE,
                                   confidence_step_down=CONFIDENCE_STEP_DOWN,
                                   confidence_step_up=CONFIDENCE_STEP_UP,
                                   max_iterations=MAX_ITERATIONS_STAMP_FINDING):
    """
    Adjust the confidence threshold dynamically to meet the target number of stamps found.

    :param imgs: List of image objects to be processed.
    :param initial_min_confidence: Initial minimum confidence for detecting stamps.
    :param target_stamps: Target number of stamps to find.
    :param confidence_step_down: Step to adjust the min_confidence by on each iteration.
    :param max_iterations: Maximum number of iterations to try adjusting confidence.
    :return: Tuple containing the adjusted min_confidence and pages with stamps.
    """
    min_confidence = initial_min_confidence
    iteration = 0
    print(f"Iteration 0: min_confidence: {min_confidence}, stamps to find: {target_stamps}")
    while iteration < max_iterations:
        pages_with_stamp = find_stamp_pages(imgs, aws_rekognition_model_id, min_confidence, target_stamps)
        found_stamps = len(pages_with_stamp)
        print(f"Iteration {iteration + 1}, min_confidence: {min_confidence}, stamps found: {found_stamps}")
        if found_stamps == target_stamps:
            return min_confidence, pages_with_stamp
        elif found_stamps < target_stamps:
            # Increase confidence if fewer stamps than expected are found
            min_confidence -= confidence_step_down
        else:
            # Decrease confidence if more stamps than expected are found
            min_confidence += confidence_step_up
        # Avoid invalid confidence values
        min_confidence = max(10, min(min_confidence, 100))
        iteration += 1
    print("Reached maximum iterations or could not meet target stamps count within confidence bounds.")
    return min_confidence, pages_with_stamp


def find_stamp_pages(imgs, aws_model_id, min_confidence, target_stamps):
    """
    Find the pages with stamps and stop processing once the target number of stamped pages is found.

    Parameters:
    - imgs: List of images to process.
    - aws_model_id: AWS model ID for rekognition.
    - min_confidence: Minimum confidence level for detecting stamps.
    - target_stamps: Target number of stamps to find before stopping the search.

    Returns:
    - List of indices for pages with stamps.
    """
    pages_with_stamp = []
    for page, image in enumerate(imgs):
        image_bytes = image_to_bytes(image)  # Getting byte data from buffer
        try:
            response = rekognition.detect_custom_labels(
                ProjectVersionArn=aws_model_id,
                Image={'Bytes': image_bytes},
                MinConfidence=min_confidence,
            )

            if response["CustomLabels"]:
                print(f"Page {page + 1}:")
                for label in response["CustomLabels"]:
                    print(f"  Label: {label['Name']}, Confidence: {label['Confidence']:.2f}%")
                pages_with_stamp.append(page)

                # Check if we've found the target number of stamps
                if len(pages_with_stamp) >= target_stamps:
                    print(f"Target of {target_stamps} stamps found. Stopping early.")
                    break
            else:
                print(f"Page {page + 1}: No custom labels found.")
        except Exception as e:
            print(f"Error processing page {page}: {e}")

    return pages_with_stamp


def ordering_result(result,
                    page,
                    pages_with_stamp,
                    min_confidence=MIN_CONFIDENCE_LAMBDA):
    print(f"Page {page + 1}:")
    page_confidences = []
    for label in result['predictions']:
        if label['score'] >= min_confidence:
            print(f" Label: Stamp, Confidence: {label['score']:.2f}")
            page_confidences.append(label['score'])
    if page_confidences:
        pages_with_stamp.append(page)
    return pages_with_stamp


def find_stamp_pages_lambda(imgs,
                            max_workers=MAX_WORKERS,
                            min_confidence=MIN_CONFIDENCE_LAMBDA):
    """
    Find the pages containing stamps using AWS Lambda and processing images in parallel.

    Parameters:
    - imgs: List of images to process.
    - min_confidence: Minimum confidence level for stamp detection.
    - max_workers: Maximum number of workers for parallel AWS Lambda calls.

    Returns:
    - List of indices for pages containing stamps.
    """
    pages_with_stamp = []

    def invoke_lambda(page, image):
        image_base64 = image_to_base64(image)
        payload = {'body': image_base64, 'page': page}
        try:
            response = lambda_client.invoke(
                FunctionName='detector_de_sello_new',
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            result = json.loads(response['Payload'].read().decode('utf-8'))
            result = json.loads(result['body'])
            print(result)
            return page, result, response['StatusCode']
        except Exception as e:
            print(f"Error processing page {page}: {e}")
            return page, None, None

    # Crear un pool de hilos para invocar las funciones Lambda en paralelo
    # Ajusta este valor según tus límites de concurrencia
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Enviar las tareas al pool de hilos
        futures = [executor.submit(invoke_lambda, page, image) for page, image in enumerate(imgs)]

        # Esperar a que todas las tareas se completen y manejar errores
        for future in as_completed(futures):
            page, result, status_code = future.result()
            if result is not None:
                if result['predictions']:
                    pages_with_stamp = ordering_result(result,
                                                       page,
                                                       pages_with_stamp,
                                                       min_confidence)
                else:
                    print(f"Page {page + 1}: No custom labels found.")
            else:
                try:
                    if status_code == 202:
                        (page,
                         result,
                         processing_time,
                         status_code) = invoke_lambda(page,
                                                      imgs[page])
                        if result['predictions']:
                            pages_with_stamp = ordering_result(result,
                                                               page,
                                                               pages_with_stamp,
                                                               min_confidence)
                        else:
                            print(f"Page {page + 1}: No custom labels found.")
                except Exception as e:
                    print(f"Error processing page {page}: {e}")
        # Crear nuevas listas ordenadas utilizando los índices ordenadas
        pages_with_stamp.sort()

    return pages_with_stamp


def ordering_result_eval(pages_with_stamp,
                         stamps_with_highscore,
                         processing_times,
                         confidences,
                         result,
                         page,
                         processing_time,
                         min_confidence=MIN_CONFIDENCE_LAMBDA):
    processing_times.append(processing_time)
    print(f"Page {page + 1}:")
    page_confidences = []
    for label in result['predictions']:
        if label['score'] >= min_confidence:
            print(f" Label: Stamp, Confidence: {label['score']:.2f}")
            page_confidences.append(label['score'])
            stamps_with_highscore.append(page)
    if page_confidences:
        pages_with_stamp.append(page)
        confidences.append(page_confidences)
    return pages_with_stamp, stamps_with_highscore, processing_times, confidences


def find_stamp_pages_lambda_eval(imgs,
                                 min_confidence=MIN_CONFIDENCE_LAMBDA,
                                 max_workers=MAX_WORKERS):
    """
    Find the pages containing stamps using AWS Lambda and processing images in parallel.

    Parameters:
    - imgs: List of images to process.
    - min_confidence: Minimum confidence level for stamp detection.
    - max_workers: Maximum number of workers for parallel AWS Lambda calls.

    Returns:
    - pages_with_stamp: List of indices for pages containing stamps.
    - stamps_with_highscore: List of stamps with high confidence.
    - processing_times: List of the duration of each lambda call.
    - confidences: List of the confidence of each stamp with high confidence.
    """
    pages_with_stamp = []
    stamps_with_highscore = []
    processing_times = []
    confidences = []

    def invoke_lambda(page, image):
        image_base64 = image_to_base64(image)
        payload = {'body': image_base64, 'page': page}
        try:
            start_time = time.time()
            response = lambda_client.invoke(
                FunctionName='detector_de_sello_new',
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            result = json.loads(response['Payload'].read().decode('utf-8'))
            result = json.loads(result['body'])
            print(result)
            end_time = time.time()
            processing_time = end_time - start_time
            return page, result, processing_time, response['StatusCode']
        except Exception as e:
            print(f"Error processing page {page}: {e}")
            return page, None, None, None

    # Crear un pool de hilos para invocar las funciones Lambda en paralelo
    # # Ajusta este valor según tus límites de concurrencia
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Enviar las tareas al pool de hilos
        futures = [executor.submit(invoke_lambda, page, image) for page, image in enumerate(imgs)]

        # Esperar a que todas las tareas se completen y manejar errores
        for future in as_completed(futures):
            page, result, processing_time, status_code = future.result()
            if result is not None:
                if result['predictions']:
                    (pages_with_stamp,
                     stamps_with_highscore,
                     processing_times,
                     confidences) = ordering_result_eval(pages_with_stamp,
                                                         stamps_with_highscore,
                                                         processing_times,
                                                         confidences,
                                                         result,
                                                         page,
                                                         processing_time,
                                                         min_confidence)
                else:
                    print(f"Page {page + 1}: No custom labels found.")
            else:
                try:
                    if status_code == 202:
                        (page,
                         result,
                         processing_time,
                         status_code) = invoke_lambda(page,
                                                      imgs[page])
                        if result['predictions']:
                            (pages_with_stamp,
                             stamps_with_highscore,
                             processing_times,
                             confidences) = ordering_result_eval(pages_with_stamp,
                                                                 stamps_with_highscore,
                                                                 processing_times,
                                                                 confidences,
                                                                 result,
                                                                 page,
                                                                 processing_time,
                                                                 min_confidence)
                        else:
                            print(f"Page {page + 1}: No custom labels found.")
                except Exception as e:
                    print(f"Error processing page {page}: {e}")
        sorted_indices = sorted(range(len(pages_with_stamp)), key=lambda i: pages_with_stamp[i])

        # Crear nuevas listas ordenadas utilizando los índices ordenadas
        pages_with_stamp.sort()
        stamps_with_highscore.sort()
        sorted_confidences = [confidences[i] for i in sorted_indices]
        sorted_processing_times = [processing_times[i] for i in sorted_indices]

    return pages_with_stamp, stamps_with_highscore, sorted_processing_times, sorted_confidences


def separar_paginas_en_oficios(paginas, oficios_inicios):
    """Separa las paginas en oficios basado en los inicios de cada oficio.

    Args:
        paginas (list): Lista de todas las paginas.
        oficios_inicios (list): Indices de las paginas donde comienza cada oficio.

    Returns:
        list: Lista de oficios, donde cada oficio es un string de paginas concatenadas.
    """
    oficios = []
    # Itera sobre los inicios de los oficios
    for j in range(len(oficios_inicios)):
        inicio = oficios_inicios[j]
        # Si no es el último oficio, establece el fin como el inicio del siguiente oficio
        if j < len(oficios_inicios) - 1:
            fin = oficios_inicios[j + 1]
        else:
            # Para el último oficio, toma todas las páginas hasta el final
            fin = len(paginas)
        # Concatena las páginas del oficio actual y las añade a la lista de oficios
        oficios.append(paginas[inicio:fin])
    return oficios
