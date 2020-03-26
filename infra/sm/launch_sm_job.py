# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from sagemaker import get_execution_role
import sagemaker as sage
from sagemaker.estimator import Estimator
from sagemaker.inputs import FileSystemInput
import datetime
import subprocess
import sys

def get_str(cmd):
    content = subprocess.check_output(cmd, shell=True)
    return str(content)[2:-3]

account = get_str("echo $(aws sts get-caller-identity --query Account --output text)")
region = get_str("echo $(aws configure get region)")

s3_bucket  =   'smart-invoice'
prefix = "train" 
s3train = f's3://{s3_bucket}/{prefix}'
train = sage.session.s3_input(s3train, distribution='FullyReplicated', 
                        content_type='application/tfrecord', s3_data_type='S3Prefix')
s3_output_location = f's3://{s3_bucket}/{prefix}/sagemaker_training_release'
data_channels = {'train': train}

image = str(sys.argv[1])
sess = sage.Session()
image_name=f"{account}.dkr.ecr.{region}.amazonaws.com/{image}"
sagemaker_iam_role = str(sys.argv[2])
num_gpus = 1
num_nodes = 1
instance_type = 'ml.p2.xlarge'
custom_mpi_cmds = []

job_name = "maskrcnn-{}x{}-{}".format(num_nodes, num_gpus, image)

output_path = s3_output_location

# lustre_input = FileSystemInput(file_system_id='fs-03f556d03c3c590a2',
#                                file_system_type='FSxLustre',
#                                directory_path='/fsx',
#                                file_system_access_mode='ro')

hyperparams = {
    "sagemaker_use_mpi": "True",
    "sagemaker_process_slots_per_host": num_gpus,
    "num_gpus":num_gpus,
    "num_nodes": num_nodes,
    "custom_mpi_cmds": custom_mpi_cmds,
    "mode_fpn": "True",
    "mode_mask": "True",
    "eval_period": 1,
    "batch_norm": "FreezeBN"
}

metric_definitions=[      
            {
            "Name": "maskrcnn_loss/accuracy",
            "Regex": ".*maskrcnn_loss/accuracy:\\s*(\\S+).*"
        },
        {
            "Name": "maskrcnn_loss/fg_pixel_ratio",
            "Regex": ".*maskrcnn_loss/fg_pixel_ratio:\\s*(\\S+).*"
        },
        {
            "Name": "maskrcnn_loss/maskrcnn_loss",
            "Regex": ".*maskrcnn_loss/maskrcnn_loss:\\s*(\\S+).*"
        },
        {
            "Name": "maskrcnn_loss/pos_accuracy",
            "Regex": ".*maskrcnn_loss/pos_accuracy:\\s*(\\S+).*"
        },
        {
            "Name": "mAP(bbox)/IoU=0.5",
            "Regex": ".*mAP\\(bbox\\)/IoU=0\\.5:\\s*(\\S+).*"
        },
        {
            "Name": "mAP(bbox)/IoU=0.5:0.95",
            "Regex": ".*mAP\\(bbox\\)/IoU=0\\.5:0\\.95:\\s*(\\S+).*"
        },
        {
            "Name": "mAP(bbox)/IoU=0.75",
            "Regex": ".*mAP\\(bbox\\)/IoU=0\\.75:\\s*(\\S+).*"
        },
        {
            "Name": "mAP(bbox)/large",
            "Regex": ".*mAP\\(bbox\\)/large:\\s*(\\S+).*"
        },
        {
            "Name": "mAP(bbox)/medium",
            "Regex": ".*mAP\\(bbox\\)/medium:\\s*(\\S+).*"
        },
        {
            "Name": "mAP(bbox)/small",
            "Regex": ".*mAP\\(bbox\\)/small:\\s*(\\S+).*"
        }
            
        
]
               

estimator = Estimator(image_name, role=sagemaker_iam_role, output_path=output_path,
                      train_instance_count=num_nodes,
                      train_instance_type=instance_type,
                      sagemaker_session=sess,
                      train_volume_size=200,
                      base_job_name=job_name,                      
                      hyperparameters=hyperparams)

estimator.fit(inputs=data_channels, logs=True)
