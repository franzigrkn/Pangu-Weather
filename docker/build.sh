#!/bin/bash
set -e

# Image name and tag
IMAGE_NAME="panguweather-base-new"
IMAGE_TAG="0.0.0-dev"
export IMAGE_NAME
export IMAGE_TAG

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker build \
  --file ${DIR}/Dockerfile \
  --secret id=netrc,src=$HOME/.netrc \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  --tag gitlab-master.nvidia.com:5005/dvl/panguweather:main \
  ${DIR}/..


if [ "$1" == "--push" ]; then
  docker push gitlab-master.nvidia.com:5005/dvl/panguweather:main
fi