import os
import traceback
from pandas.io.formats.style import Styler
from pandas import ExcelWriter
from typing import Any
from PIL import Image
from io import BytesIO
import logging
from circulares_info_extraction.circular import Circular
from circulares_info_extraction.api_clients import dynamodb_resource, s3_client
from time import time
from boto3.dynamodb.conditions import Key
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

TABLE_NAME = os.environ.get("TABLE_NAME")
S3_RESULTS_BUCKET = os.environ.get("S3_RESULTS_BUCKET")
S3_TOPROCESS_BUCKET = os.environ.get("S3_TOPROCESS_BUCKET")


# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_tiff_image(file_content: bytes, filename: str, tipo_circular: str) -> Any:
    try:
        image_stream = BytesIO(file_content)
        nombre_circular = filename.split("/")[-1].rstrip(".tif")
        with Image.open(image_stream) as imagen_tiff_circular:
            circular = Circular(imagen_tiff_circular, nombre_circular, tipo_circular)
            start_time = time()

            if tipo_circular == "retencion_suspension_remision":
                logger.debug("Process circular retencion/suspension/remision")
                circular.process_circular_retencion_suspension_remision()
            elif tipo_circular == "informativa":
                logger.debug("Process circular informativa")
                circular.process_circular_informativa()
            elif tipo_circular == "normativa":
                logger.debug("Process circular normativa")
                circular.process_circular_normativa()
            elif tipo_circular == "combinada":
                logger.debug("Process circular combinada")
                circular.classify_and_process_circular()

            end_time = time()
            logger.debug(f"Tiempo de procesamiento: {end_time - start_time} segundos")
            resultado_df_validated = circular.results_df_validated
            time_to_process = int(end_time - start_time)
    
    except Exception as e:
        # Registrar el error completo con traceback
        logger.error(f"Error procesando la imagen: {str(e)}")
        traceback.print_exc()

        # Retornar "error" y None para indicar el fallo
        resultado_df_validated = None
        time_to_process = None

    return resultado_df_validated, time_to_process

def save_dataframe_to_s3(dataframe_style: Styler, s3_bucket: str, s3_key: str) -> None:
    buffer = BytesIO()
    with ExcelWriter(buffer) as writer:
        dataframe_style.to_excel(writer, sheet_name="Hoja1", index=False)
    buffer.seek(0)

    # Guardar el resultado en S3
    s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=buffer.getvalue())

def save_result_to_dynamoands3(filename, resultado_df_validated, time_to_process):
    table = dynamodb_resource.Table(TABLE_NAME)
    
    # Verificar si el ítem con ese Filename ya existe usando get_item
    response = table.get_item(Key={'Filename': filename})
    item = response.get('Item')

    if item:
        # Si el ítem existe, obtener el timestamp
        timestamp = item['Timestamp']
        
        if resultado_df_validated is None:
            # Actualizar el ítem con el estado de error
            table.update_item(
                Key={'Filename': filename},
                UpdateExpression="SET #st = :status",
                ExpressionAttributeNames={"#st": "Status"},
                ExpressionAttributeValues={":status": "error"}
            )
        else:
            # Generar la clave para el objeto en S3 y guardar el DataFrame en S3
            s3_key = f"{filename}.xlsx"
            save_dataframe_to_s3(resultado_df_validated, S3_RESULTS_BUCKET, s3_key)

            # Actualizar el ítem con el nuevo estado, clave de S3 y tiempo de procesamiento
            table.update_item(
                Key={'Filename': filename},
                UpdateExpression="SET #st = :status, #s3 = :s3_key, #tp = :timetoprocess",
                ExpressionAttributeNames={
                    "#st": "Status",
                    "#s3": "S3Key",
                    "#tp": "TimeToProcess"
                },
                ExpressionAttributeValues={
                    ":status": "completed",
                    ":s3_key": s3_key,
                    ":timetoprocess": time_to_process
                }
            )
    else:
        print(f"No se encontró ningún ítem con Filename: {filename}")

def check_and_save_to_dynamo(filename):
    # Obtener el timestamp UTC actual
    utc_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Conectar con la tabla de DynamoDB
    table = dynamodb_resource.Table(TABLE_NAME)

    # Verificar si el ítem con ese filename ya existe
    response = table.get_item(
        Key={'Filename': filename}
    )

    if 'Item' in response:
        print(f"El archivo {filename} ya existe en la tabla.")
        return False
    else:
        # Si no existe, agregar el nuevo ítem
        table.put_item(
            Item={
                'Status': 'in progress',
                'Filename': filename,
                'Timestamp': utc_time
            }
        )
        print(f"El archivo {filename} se ha guardado en la tabla.")
        return True

def main():
    # Extraer la key desde las variables de entorno pasado por eventbridge
    TABLE_NAME = os.environ.get("TABLE_NAME")       

    if not s3_key:
        logger.error("Bucket or Key missing from environment variables.")
        return

    # Obtener los metadatos del objeto en S3
    response = s3_client.head_object(Bucket=S3_TOPROCESS_BUCKET, Key=s3_key)
    metadata = response['Metadata']
    filename = metadata.get('filename')
    circular_type = metadata.get('circular_type')
    
    if check_and_save_to_dynamo(filename):
        # Descargar el archivo desde S3
        s3_response = s3_client.get_object(Bucket=S3_TOPROCESS_BUCKET, Key=s3_key)
        file_content = s3_response['Body'].read()

        # Procesar la imagen
        resultado_df_validated, time_to_process = process_tiff_image(file_content, s3_key, circular_type)

        # Guardar el resultado en S3 y DynamoDB
        save_result_to_dynamoands3(filename, resultado_df_validated, time_to_process)

if __name__ == "__main__":
    main()
