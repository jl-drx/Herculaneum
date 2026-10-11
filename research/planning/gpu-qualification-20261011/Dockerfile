# Provider digest recorded in RUNTIME_RECEIPT.json. This image has not been run here.
FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime@sha256:00ff90463ea886af84a8501ad0299fe29db47d110131c21c2fc101a2788dbda9
WORKDIR /opt/plinius-qualification
COPY native_model.py benchmark.py ./
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENTRYPOINT ["python", "benchmark.py"]
