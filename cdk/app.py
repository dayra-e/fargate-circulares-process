from aws_cdk import App, Environment

from stacks.fargate_stack import FargateStack
from stacks.dynamo_and_s3_stack import DynamoS3Stack

app = App()
region = "eu-west-2"
env_eu_west = Environment(region=region)

# Crear el stack de DynamoDB
dynamo_stack = DynamoS3Stack(app, 
                             f"DynamoS3Stack-{region}",
                             env=env_eu_west)

# Crear el stack de Fargate y pasar el ARN de la tabla DynamoDB
fargate_stack = FargateStack(app, 
                             f"FargateStack-{region}",
                             env=env_eu_west, 
                             table_arn=dynamo_stack.table_arn,
                             bucket_results_arn=dynamo_stack.bucket_results_arn)

app.synth()