from aws_cdk import (Stack,  
                     aws_ec2 as ec2)
from constructs import Construct

class VpcStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        # Crear una VPC con subnets públicas
        self.vpc = ec2.Vpc(self, "FargateVPC", 
                      max_azs=2,  # Usar 2 zonas de disponibilidad
                      subnet_configuration=[
                          ec2.SubnetConfiguration(
                              name="PublicSubnet",
                              subnet_type=ec2.SubnetType.PUBLIC  # Subnet pública con IP pública
                          )            
                      ])
        
        self.vpc.add_gateway_endpoint("S3Endpoint",
        service=ec2.GatewayVpcEndpointAwsService.S3
        )

        self.vpc.add_gateway_endpoint("DynamoDbEndpoint",
            service=ec2.GatewayVpcEndpointAwsService.DYNAMODB
        )


        self.vpc.add_interface_endpoint("TextractEndpoint",
        service=ec2.InterfaceVpcEndpointAwsService.TEXTRACT
        )
        
        self.vpc.add_interface_endpoint("LambdaEndpoint",
        service=ec2.InterfaceVpcEndpointAwsService.LAMBDA_
        )
        
        self.vpc_output = self.vpc