FROM nvcr.io/nvidia/pytorch:23.06-py3

ENV LC_ALL="C.UTF-8" LESSCHARSET="utf-8"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace/working

RUN apt update && apt upgrade -y \
    && DEBIAN_FRONTEND=nointeractivetzdata \
    TZ=Asia/Tokyo \
    apt install -y \
    make \
    cmake \
    unzip \
    git \
    curl \
    wget \
    tzdata \
    locales \
    sudo \
    tar \
    gcc \
    g++ \
    libgl1-mesa-dev \
    software-properties-common

