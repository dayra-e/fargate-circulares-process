from aws_cdk import (Stack,  
                     aws_ec2 as ec2)
from constructs import Construct

class SecurityStack(Stack):
    def __init__(self, scope: Construct, id: str, vpc: ec2.Vpc, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
                
        # Crear un Security Group sin tráfico entrante permitido
        self.sg_fargate = ec2.SecurityGroup(self, "FargateSecurityGroup",
            vpc=vpc,
            description="Security Group for Fargate, no inbound traffic",
            allow_all_outbound=False  # Permitir todo el tráfico saliente
        )

        # No se permite tráfico entrante
        # Permitir tráfico HTTPS saliente (puerto 443) a API externas como ChatGPT
        self.sg_fargate.add_egress_rule(
            peer=ec2.Peer.any_ipv4(),  # Permitir tráfico saliente hacia cualquier dirección
            connection=ec2.Port.tcp(443),  # Puerto HTTPS (443)
            description="Allow outbound HTTPS traffic to external APIs"
        )

        self.sg_output = self.sg_fargate