ARG RAPIDS_VERSION=23.12
ARG CUDA_VERSION=12.0
ARG PYTHON_VERSION=3.10
FROM nvcr.io/nvidia/rapidsai/base:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-py${PYTHON_VERSION} as src

USER root

RUN apt update && \
    apt install -y unzip

WORKDIR /sfcc

ADD . .

RUN for zip_file in $(ls *.zip); do unzip ${zip_file} && rm ${zip_file}; done


FROM nvcr.io/nvidia/rapidsai/base:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-py${PYTHON_VERSION}

USER root

RUN apt update && \
    apt upgrade -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists* /var/cache/apt/*

USER rapids:conda

WORKDIR ./sfcc

COPY --chown=rapids:conda --chmod=644 --from=src /sfcc .

RUN pip3 install -r requirements.txt

CMD ["python3", "sfcc_prediction.py"]
