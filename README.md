# Open Cities AI Challenge
A not very impressive attempt at the [DrivenData OpenCities AI Challenge](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/). Put together a deep learning model based on ResNet50 that did passably well (IoU = 70% or so), but limited training because AWS P3 instances aren't cheap! Luckily I won a consolation prize by writing a [blog post on responsible AI](https://rdrn.me/ethics-ai/)!

Based on: [Open Cities AI Challenge Benchmark Model](https://github.com/azavea/open-cities-ai-challenge-benchmark-model)  
AWS setup: [Raster Vision AWS repo](https://github.com/azavea/raster-vision-aws)

## Setup
Build the Docker image. This will create a local Docker image called 'raster-vision-wb-africa' that includes the code within the `experiment` module, which you will need to run this experiment.
```
./docker/build
```

If you would like to run the workflow on remotely you will need to publish the image to AWS Batch.
Edit `docker/publish_image` to reference the ECR cpu and gpu repos that you created during the [Raster Vision AWS setup](https://github.com/azavea/raster-vision-aws#raster-vision-aws-batch-runner-setup). Both images will be tagged with `world-bank-challenge`. When/if you make changes to the aux commands you will need to repeat this process (i.e. rebuild the Docker image and publish the updated version to ecr). Publish your docker image to both gpu and cpu repos:
```
./docker/publish_image
```


Run the Docker container with the `run` script:
```
./docker/run --aws
```

## Preprocess
Split training scenes into smaller files to fit into memory.
```
EXP_ID=<exp_id> ./runner split
```

## Test
Test a subset of the imagery on local machine:
```
EXP_ID=<exp_id> ./runner test
```

## Train, predict & postprocess
The model training and prediction configuration is located in `experiment/experiment.py`.This will submit a series of jobs to AWS Batch and print out a summary of each, complete with an outline of which task must finish before the job in question can start. If you would like to first do a 'dry run' (i.e. see the aforementioned output without actually submitting any jobs), add `-n` to the end of the command. Use the 'test' flag (`-a test True`) to run an experiment on a small subset of the data and with very short training times. This will not yield useful predictions but may be helpful to make sure everything is configured correctly before trying to run the full experiment. This includes a step to convert `2` (used by rv for background) to `0` (used by competition rules).
```
EXP_ID=<exp_id> ./runner batch
```

## Evaluate
The training stage should take roughly 9 hours running on batch. Once training is done, prediction, postprocessing and evaluation will complete shortly after. You can see how the model performed on the validation set by looking at the output of eval (`<root_uri>/eval/<exp_id>/eval.json`). You can also view predictions (`<root_uri>/predict/<exp_id>/<scene id>.tif`) and compare to the original images.

## Submit
Zip results then download and submit:
```
EXP_ID=<exp_id> ./runner zip
aws s3 cp s3://carderne-rv/final/<exp_id>.zip . --profile rv
```
