#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Dir:" ${DIR}

bash ${DIR}/../docker/build.sh

docker build \
  --file ${DIR}/Dockerfile \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --secret id=ngc,src=${HOME}/.ngc/config  \
  --secret id=netrc,src=${HOME}/.netrc  \
  --secret id=cdsapirc,src=${HOME}/.cdsapirc  \
  --tag panguweather-dev:${USER} ${DIR}/..

DEV_CONTAINER_OUTPUT_DIR="${DEV_CONTAINER_OUTPUT_DIR:-/var/tmp/panguweather}"
DEV_CONTAINER_DATA_DIR="${DEV_CONTAINER_DATA_DIR:-/var/tmp/data}"
DEV_CONTAINER_GPUS="${DEV_CONTAINER_GPUS:-all}"

# Create data and output directory
mkdir -p $DEV_CONTAINER_DATA_DIR
mkdir -p $DEV_CONTAINER_OUTPUT_DIR/outputs

if [ "$1" == "--build-only" ]; then
  exit 0
fi

# Following NVIDIA's container recommendations
docker run --interactive --tty --rm \
    --gpus $DEV_CONTAINER_GPUS \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --volume $DEV_CONTAINER_DATA_DIR:/data \
    --volume $DEV_CONTAINER_OUTPUT_DIR:/workspace \
    --volume ${DIR}/..:/panguweather \
    --env-file ${DIR}/../scripts/env.env \
    panguweather-dev:${USER} /bin/bash
