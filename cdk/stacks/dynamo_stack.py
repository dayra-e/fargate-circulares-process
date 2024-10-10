from aws_cdk import (
    Stack,
    aws_dynamodb as dynamodb,
)
from constructs import Construct

class DynamoStack(Stack):

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
        
        self.table_arn = table.table_arn