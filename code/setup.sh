#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=raghudv_resnet_project      #Set the job name to "JobExample4"
#SBATCH --time=14:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=32                   #Request 1 task
#SBATCH --mem=32G                  #Request 2560MB (2.5GB) per node
#SBATCH --output=errOut.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:2                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456             #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=raghudv@tamu.edu    #Send all emails to email_address

python main.py
