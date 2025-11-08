---
title: How you can run the same Python distribution on EKS and AWS Lambda (using UV)
description: How I solved Python dependency issues when deploying the same ML model to both Kubernetes and AWS Lambda using awslambdaric.
date: 2025-11-08
tags: posts
---

I recently tackled a problem that ate up a few hours of my day, and I'm documenting the solution for anyone who hits this same wall in the future.

## The Setup

I'm doing my development with `uv`, and I was working on a microservice that serves an ML model via API. It's a FastAPI application running at on an EKS cluster in Kubernetes. Everything was working great.

Then came another requirement: we needed to use the exact same model inference pipeline in an AWS Lambda function. "Easy peasy," I thought. "I'll just pack the whole thing in another Docker container and ship it to Lambda."

## The Problem

I packaged everything up, but immediately hit a wall with `uv` installations. The exact same packages that worked perfectly in my FastAPI application were causing chaos in Lambda. Specifically, I kept running into issues with `numpy`, `pyarrow`, and a few other scientific computing packages.

After some digging, I found the problem:

- **FastAPI app**: Used the standard `public.ecr.aws/docker/library/python:3.11-slim` image
- **Lambda app**: Used `public.ecr.aws/lambda/python:3.11` image

These two Python distributions are different enough that my dependencies were breaking left and right. I tried fixing the dependency issues manually, but it was a nightmare.

## Why Can't I Just Use Regular Python for Lambda?

Here's the thing about AWS Lambda: it's not just running your code on some random server. Lambda has a specific runtime architecture designed for serverless execution.

When a Lambda function is invoked, AWS needs to:
1. Find (or spin up) an execution environment
2. Load your code and dependencies
3. Execute your handler function
4. Return the result

This whole process, especially when starting from a cold environment (called a "cold start"), needs to be fast and predictable. AWS Lambda's Python distribution includes specific system libraries, dependencies, and configurations that are optimized for this runtime environment.

The official `public.ecr.aws/lambda/python:3.11` base image includes:
- The Lambda Runtime Interface Client
- System-level dependencies that Lambda expects
- Proper file system structure (`/var/task`, `/opt`, etc.)
- Libraries compiled against Amazon Linux 2

If you use a standard Python image, you're essentially trying to fit a square peg in a round hole. Your packages might compile against different system libraries, leading to cryptic runtime errors or, worse, subtle bugs that only show up under load.

(you can read more about lambda's in [AWS Lambda Architecture Deep Dive](https://medium.com/@joudwawad/aws-lambda-architecture-deep-dive-bef856b9b2c4))

## The Solution: awslambdaric
Ive started to think of making a new package within my repository with custom uv requirements, but then you will need to train the same model with different packaging versions, and you will be in a mess.

Eventually, I discovered `awslambdaric` (AWS Lambda Runtime Interface Client), which turned out to be the perfect middle ground. This package lets you use a standard Python distribution while still being compatible with Lambda's runtime.

The beauty of `awslambdaric` is that it implements the Lambda Runtime API, so your container can communicate with Lambda's execution environment without needing the full AWS Lambda base image. The only downside? Local testing requires a bit more setup, but it's totally manageable.


Here's my original Dockerfile for the FastAPI application:
```dockerfile
FROM public.ecr.aws/docker/library/python:3.11-slim AS base
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

FROM base AS app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY application.py ./
COPY src/ ./src/
COPY models/ ./models/
RUN uv sync --frozen
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--loop", "uvloop"]
```

And here's the new Lambda-compatible Dockerfile using `awslambdaric`:
```dockerfile
ARG FUNCTION_DIR="/function"

FROM python:3.11 AS build-image
ARG FUNCTION_DIR
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN mkdir -p ${FUNCTION_DIR}
COPY pyproject.toml uv.lock ${FUNCTION_DIR}/
RUN uv pip install --target ${FUNCTION_DIR} --no-cache -r ${FUNCTION_DIR}/pyproject.toml
RUN pip install --target ${FUNCTION_DIR} awslambdaric

COPY application.py ${FUNCTION_DIR}/
COPY src/ ${FUNCTION_DIR}/src/
COPY lambda_aws/ ${FUNCTION_DIR}/lambda_aws/
COPY models/ ${FUNCTION_DIR}/models/

FROM public.ecr.aws/docker/library/python:3.11-slim AS app
ARG FUNCTION_DIR
WORKDIR ${FUNCTION_DIR}
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["lambda_aws.lambda_function.handler"]
```

Key differences:
- Using a multi-stage build to install dependencies in the first stage
- Installing `awslambdaric` alongside other dependencies
- Setting the entrypoint to run Python with the `awslambdaric` module
- Pointing to your Lambda handler function in the CMD

## Testing Locally

Testing Lambda containers locally requires the AWS Lambda Runtime Interface Emulator (RIE). Here's how to set it up:

```bash
# Build Lambda image
docker build -f lambda-dockerfile \
  -t your-project-name .

# Download AWS Lambda Runtime Interface Emulator (one-time setup)
mkdir -p ~/.aws-lambda-rie && \
  curl -Lo ~/.aws-lambda-rie/aws-lambda-rie \
  https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie && \
  chmod +x ~/.aws-lambda-rie/aws-lambda-rie

# Run Lambda container with RIE
docker run \
  -v ~/.aws-lambda-rie:/aws-lambda \
  -p 9000:8080 \
  --entrypoint /aws-lambda/aws-lambda-rie \
  --env-file .env \
  your-project-name-lambda \
  /usr/local/bin/python -m awslambdaric lambda_aws.lambda_function.handler

# Test with a sample invocation
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d 'your payload'
```

The RIE acts as a local Lambda service, letting you invoke your function just like AWS would, without deploying anything to the cloud.

