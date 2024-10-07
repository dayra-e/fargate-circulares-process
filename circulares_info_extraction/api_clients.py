import boto3
from botocore.config import Config
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
# Configure the connection pool size
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION")
OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")

config = Config(
    region_name=AWS_REGION,
    max_pool_connections=200  # Increase the connection pool size
)

# AWS
rekognition = boto3.client('rekognition', region_name=AWS_REGION)
lambda_client = boto3.client('lambda', config=config)
textract_client = boto3.client('textract', config=config)
dynamodb_resource = boto3.resource('dynamodb', region_name=AWS_REGION)  # Aquí configuramos dynamodb_resource correctamente
s3_client = boto3.client('s3', config=config)
# OPEN AI
openai_client = OpenAI(api_key=OPEN_AI_KEY)