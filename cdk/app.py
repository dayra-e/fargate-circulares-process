from aws_cdk import App, Environment

from stacks.fargate_stack import FargateStack
from stacks.dynamo_stack import DynamoStack
from stacks.s3_stack import S3Stack
from stacks.security_stack import SecurityStack
from stacks.vpc_stack import VpcStack
from stacks.ecr_stack import ECRStack


app = App()
region = "eu-west-2"
env_eu_west = Environment(region=region)

# Crear el stack de DynamoDB
dynamo_stack = DynamoStack(app, 
                             f"DynamoS3Stack-{region}",
                             env=env_eu_west)
# Crear el stack de S3
s3_stack = S3Stack(app, 
                   f"S3Stack-{region}",
                   env=env_eu_west)
ecr_stack = ECRStack(app,
                    f"ECRStack-{region}",
                    env=env_eu_west)
vpc_stack = VpcStack(app, 
                     f"VPCStack-{region}",
                     env=env_eu_west)
security_stack = SecurityStack(app, 
                               f"SecurityStack-{region}",
                               vpc=vpc_stack.vpc,
                               env=env_eu_west)

# Crear el stack de Fargate y pasar el ARN de la tabla DynamoDB
fargate_stack = FargateStack(app, 
                             f"FargateStack-{region}",
                             env=env_eu_west, 
                             table_arn=dynamo_stack.table_arn,
                             bucket_results_arn=s3_stack.bucket_results_arn,
                             bucket_to_process=s3_stack.bucket_to_process,
                             vpc=vpc_stack.vpc_output,
                             security_group=security_stack.sg_output,
                             repository=ecr_stack.repository
                             )

app.synth()