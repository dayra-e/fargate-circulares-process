from aws_cdk import (
    Stack,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    RemovalPolicy
)
from constructs import Construct

class DynamoS3Stack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        table = dynamodb.Table(self, 
                                "FargateProcessingResultsTable",
                                table_name="ProcessingResultsFargate",
                                billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
                                partition_key=dynamodb.Attribute(
                                    name="Filename",
                                    type=dynamodb.AttributeType.STRING
                                ),
                                stream=dynamodb.StreamViewType.NEW_IMAGE
        )
        bucket_results = s3.Bucket(self, 
                                    "ResultsBucket",
                                    versioned=False,
                                    removal_policy=RemovalPolicy.DESTROY,
                                    auto_delete_objects=True
                                 )
        self.table_arn = table.table_arn
        self.bucket_results_arn = bucket_results.bucket_arn