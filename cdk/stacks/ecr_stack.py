from aws_cdk import (
    Stack,
    aws_ecr as ecr,
    RemovalPolicy,
)
from constructs import Construct

class ECRStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        self.repository = ecr.Repository(self,
                        "FargateCircularProcessRepository",
                        repository_name="fargate-circular-process",
                        removal_policy=RemovalPolicy.DESTROY,
                        image_tag_mutability=ecr.TagMutability.MUTABLE
                        )
        