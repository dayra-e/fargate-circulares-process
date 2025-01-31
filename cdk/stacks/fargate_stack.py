from aws_cdk import (
    Stack, 
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_s3 as s3,
    aws_logs as logs,
    aws_ecs as ecs,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    RemovalPolicy)
from constructs import Construct

class FargateStack(Stack):
    def __init__(self, scope: Construct, 
                 id: str, table_arn: str, 
                 bucket_results_arn: str, 
                 bucket_to_process: s3.Bucket,    
                 vpc: ec2.IVpc, 
                 security_group: ec2.SecurityGroup,
                 repository: ecr.Repository,
                 **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        # Create ECS Cluster
        cluster = ecs.Cluster(self, "FargateCluster", 
                              vpc=vpc,
                              container_insights=True)
        
        # Create a log group
        log_group = logs.LogGroup(self, "FargateTaskLogGroup",
            log_group_name="/ecs/fargate-task",
            removal_policy=RemovalPolicy.DESTROY  # Or any other policy you prefer
        )
        # Crear una política con todos los permisos necesarios
        task_role = iam.Role(self, "FargateTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )
        task_role.add_to_policy(iam.PolicyStatement(
            actions=[
                # Permisos para S3
                "s3:GetObject",
                "s3:ListBucket",
                "s3:PutObject",
                # Permisos para DynamoDB
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
                "dynamodb:GetItem",
                # Permiso para invocar una función Lambda
                "lambda:InvokeFunction",
                # Permisos para Textract
                "textract:AnalyzeDocument",
                "textract:DetectDocumentText",
                "textract:GetDocumentAnalysis",
                # Permisos para CloudWatch Logs
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:CreateLogGroup"              
            ],
            resources=[
                # Recursos de S3
                bucket_to_process.bucket_arn,  # Bucket ARN
                f"{bucket_to_process.bucket_arn}/*",  # Objetos dentro del bucket
                bucket_results_arn,
                f"{bucket_results_arn}/*",
                # Recursos de DynamoDB
                table_arn,
                "arn:aws:lambda:eu-west-2:330797680824:function:detector_de_sello_new",
                "*",
                log_group.log_group_arn,  # Referencing the ARN of the log group created above
                f"{log_group.log_group_arn}:*"  # Optionally include streams inside the log group
            ]
        ))        
        
        # Define a Fargate task definition
        task_definition = ecs.FargateTaskDefinition(self, "FargateTaskDef",
            memory_limit_mib=6144,  # 8 GB de memoria
            cpu=1024,
            task_role=task_role,
        )
        
        container = task_definition.add_container("FargateContainer",
            image=ecs.ContainerImage.from_ecr_repository(repository),
            logging=ecs.AwsLogDriver(
                stream_prefix="FargateTask",
                log_group=log_group
            )
        )
        
        fargate_service = ecs.FargateService(self, "CircularFargateService",
            cluster=cluster,
            task_definition=task_definition,
            security_groups=[security_group], 
            assign_public_ip=True,
            desired_count=0,# Asignar una IP pública
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),  # Ejecutar en subnets públicas
        )
        
        # Configurar el Auto Scaling sin métricas adicionales
        scaling = fargate_service.auto_scale_task_count(
            min_capacity=0,
            max_capacity=6
        )        
        
        # Create EventBridge rule that triggers the Fargate task when an S3 event occurs
        rule = events.Rule(self, "EventRule",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {
                        "name": [bucket_to_process.bucket_name]
                    }
                }
            )
        )
        
        # Add ECS Fargate task as a target to the EventBridge rule, passing the S3 key as an environment variable
        rule.add_target(targets.EcsTask(
            cluster=cluster,
            task_definition=task_definition,
            subnet_selection=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            assign_public_ip=True,
            task_count=1,
            launch_type=ecs.LaunchType.FARGATE,
            retry_attempts=0,
            container_overrides=[{
                "containerName": "FargateContainer",  # Container name must match the one defined earlier
                "environment": [
                    {
                        "name": "S3_KEY",  # This is the name of the environment variable inside the container
                        "value": events.EventField.from_path("$.detail.object.key")  # This extracts the S3 key from the event
                    }
                ]
            }]
        ))