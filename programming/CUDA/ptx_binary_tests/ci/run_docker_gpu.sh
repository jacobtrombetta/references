#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(dirname "$0")"

# Source the tags from tags.sh
source "$SCRIPT_DIR/tags.sh"

# If you have a GPU instance configured in your machine
docker run -v "$SCRIPT_DIR/..":/src --gpus all --privileged -it "${image_name}" /bin/bash -l
