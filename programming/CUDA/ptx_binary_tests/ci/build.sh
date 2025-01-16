#!/bin/bash

# Source the tags from tags.sh
source "$(dirname "$0")/tags.sh"

# Navigate to the directory containing the Dockerfile
cd "$(dirname "$0")"

# Build the Docker image
docker build --build-arg CUDA=$cuda -t "${image_name}" .
