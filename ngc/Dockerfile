FROM panguweather-base-new:0.0.0-dev

# Pretty terminal colors
ENV TERM xterm-256color

WORKDIR /pangu-weather
COPY . .

ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN pip install --no-cache-dir --editable . \
    && rm -rf *.egg-info
