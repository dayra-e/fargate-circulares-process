from aws_cdk import (
    Stack,
    aws_s3 as s3,
    RemovalPolicy
)
from constructs import Construct

class S3Stack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)        
        
        bucket_results = s3.Bucket(self, 
                                    "ResultsBucket",
                                    versioned=False,
                                    removal_policy=RemovalPolicy.DESTROY,
                                    auto_delete_objects=True
                                 )
        bucket_to_process = s3.Bucket(self, 
                                        "ImagestoProcessgBucket",
                                        versioned=False,
                                        removal_policy=RemovalPolicy.DESTROY,
                                        auto_delete_objects=True
                                    )
        self.bucket_to_process_arn = bucket_to_process.bucket_arn
        self.bucket_to_process_name = bucket_to_process.bucket_name
        self.bucket_results_arn = bucket_results.bucket_arn       