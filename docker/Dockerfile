#FROM nvcr.io/nvidia/pytorch:21.10-py3
FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04
# SAL
#FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    python-is-python3 \
    python3-pip \
    # OpenCV
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_gpu.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir --upgrade setuptools
RUN python -m pip install --no-cache-dir --upgrade wheel

RUN --mount=type=secret,id=netrc,target=/root/.netrc,required=true \
    python -m pip install --no-cache-dir \
        --requirement /tmp/requirements.txt \
        # DVL group package index
        #--extra-index-url https://gitlab-master.nvidia.com/api/v4/groups/106305/-/packages/pypi/simple \
        # NCoreSDK package index
        #--extra-index-url https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/pypi/simple \
        #--find-links https://download.pytorch.org/whl/torch \
        #--find-links https://download.pytorch.org/whl/torchvision \
    && rm /tmp/requirements.txt
    
ENV DEBIAN_FRONTEND=
WORKDIR /pangu-weather