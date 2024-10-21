# fargate-circulares-process
fargate-circulares-process is a scalable, serverless application designed to automate the processing of 'circulares' documents using AWS Fargate. 
## Repository Structure
```
/fargate-circulares-process
│
├── /cdk                       # Contains AWS CDK files for deployment
│   ├── /cdk.out               # Compiled CDK output
│   └── /stacks                # CDK stacks scripts
│       └── app.py             # Main AWS CDK application
├── cdk.json                   # CDK configuration
├── circulares_info_extraction # Script for information extraction
├── .env                       # Environment variables
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── Dockerfile                 # Dockerfile for building the container image
├── LICENSE                    # License file
├── main.py                    # Main script for processing
└── README.md                  # This file
## Prerequisites
Ensure you have the following installed:
- AWS CLI
- Docker
- AWS CDK
```
## AWS Architecture
![Architecture Diagram](https://github.com/dayra-e/fargate-circulares-process/blob/main/assets/diagrama.png)
## Setup
To set up this project, follow these steps:

1. **Clone the repository:**
``` bash
git clone https://github.com/your-username/fargate-circulares-process.git 
cd fargate-circulares-process
```
2. **Set up your AWS credentials:**
``` bash
aws configure
```

## Deployment
To deploy this project on AWS Fargate, run:
``` bash
cd cdk
cdk bootstrap 
cdk deploy
```
This will set up all the necessary infrastructure on AWS using AWS CDK.
