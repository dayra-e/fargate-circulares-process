import requests
import base64
import os
import time
import boto3
import uuid
import pandas as pd
import json
import io
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime, timedelta
import pytz
from decimal import Decimal

from dotenv import load_dotenv
load_dotenv()

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION")
dynamodb = boto3.resource('dynamodb',region_name=AWS_REGION)
table_name = os.environ.get("TABLE_NAME")
table = dynamodb.Table(table_name)


S3_RESULTS_BUCKET = os.environ.get("S3_RESULTS_BUCKET")
S3_PROCESS_BUCKET = os.environ.get("S3_TOPROCESS_BUCKET")

s3_client = boto3.client('s3')


def item_by_filename(filename, s3_client, table, S3_RESULTS_BUCKET):
    # Query DynamoDB for the job status
    response = table.query(
        KeyConditionExpression=Key('Filename').eq(filename)
    )
    
    items = response.get('Items', [])
    if not items:
        return {'message': 'No process found with the given filename'}

    item = items[0]  # Assume you are interested in the first match
    status = item.get('Status', 'unknown')

    if status == 'completed':
        if 'S3Key' in item:
            s3_response = s3_client.get_object(Bucket=S3_RESULTS_BUCKET, Key=item['S3Key'])
            file_content = s3_response['Body'].read()  # Assuming binary or text data
            return {
                    'timestamp': item.get('Timestamp'),
                    'status': status,
                    'result': file_content,
                    'filename': filename,
                    'timetoprocess': item.get('TimeToProcess')
                    }
    elif status == 'in progress':
        return {'message': 'Job still in progress'}
    elif status == 'error':
        return {'message': 'Job failed'}
    else:
        return {'message': f'Unexpected job status: {status}'}



def item_by_date (date_str):
    # Convertir la fecha proporcionada al rango de tiempo correspondiente en UTC
    bolivia_tz = pytz.timezone('America/La_Paz')
    date = datetime.strptime(date_str, '%Y-%m-%d')
    start_of_day = bolivia_tz.localize(datetime.combine(date, datetime.min.time())).astimezone(pytz.utc)
    end_of_day = bolivia_tz.localize(datetime.combine(date, datetime.max.time())).astimezone(pytz.utc)
    
    # Query DynamoDB for items within the specified date range
    response = table.scan(
        FilterExpression=Attr('Timestamp').between(start_of_day.isoformat(), end_of_day.isoformat())
    )
    
    items = response.get('Items', [])
    files=[]
    for item in items:
            if item['Status'] == 'completed': 
                if 'S3Key' in item:
                    s3_response = s3_client.get_object(Bucket=S3_RESULTS_BUCKET, Key=item['S3Key'])
                    file_content = s3_response['Body'].read()
                    result= {
                            'timestamp': item['Timestamp'],
                            'status': item['Status'],
                            'result': file_content,
                            'filename': item['Filename'],
                            'timetoprocess': item['TimeToProcess']
                            }
                    files.append(result)
            elif item['Status'] == 'in progress':
                    result= {
                            'timestamp': item['Timestamp'],
                            'status': item['Status'],
                            'filename': item['Filename'],
                            }
                    files.append(result)
            elif item['Status'] == 'error':
                    result= {
                            'job_id': item['JobId'],
                            'timestamp': item['Timestamp'],
                            'status': item['Status'],
                            'filename': item['Filename'],
                            }
                    files
    if items:
        return files
    else:
        return {'message': 'No process found with the given date'}

def upload_file_to_s3(image_path, bucket_name, filename, circular_type):
    job_id = str(uuid.uuid4())
    s3_key = f"uploads/{filename}.tif"
    
    with open(image_path, "rb") as file:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=file,
            Metadata={
                'job_id': job_id,
                'filename': filename,
                'circular_type': circular_type
            }
        )
    
    return job_id, s3_key


# Parámetros de la prueba
image_path = r"C:\Projects\latam-banks-tech\bancos_app\evaluations\cases\CC-multiple_pages.tif"
filename = os.path.splitext(os.path.basename(image_path))[0]

circular_type = "combinada"

# Subir el archivo a S3 con los metadatos
s3_key = upload_file_to_s3(image_path, S3_PROCESS_BUCKET, filename, circular_type)
print(f"File uploaded to S3 with Filename: {filename}")

# Esperar 3 minutos antes de comprobar el estado
#print("Waiting for 3 minutes before checking job status...")
#time.sleep(180)  # 180 segundos = 3 minutos