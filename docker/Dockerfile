FROM python:3.11-slim-bookworm

ARG RANDOPT_COMMIT=80ec7f6d97e0e3e56b9d58fcf22094a73dd489f6

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_USE_V1=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu124 \
      --extra-index-url https://pypi.org/simple \
      torch==2.6.0 \
    && python -m pip install --no-cache-dir \
      vllm==0.8.5 \
      transformers==4.56.0 \
      numpy==1.26.4 \
      datasets==4.2.0 \
      pyarrow==25.0.1

RUN git clone https://github.com/sunrainyg/RandOpt.git /opt/RandOPT \
    && git -C /opt/RandOPT checkout --detach "$RANDOPT_COMMIT" \
    && rm -rf /opt/RandOPT/.git

WORKDIR /opt/posttrainbench0
COPY . .
RUN python -m pip install --no-deps -e . \
    && python scripts/check_environment.py \
         --data-root /opt/posttrainbench0/data/visible200 \
         --skip-gpu

ENTRYPOINT []
CMD ["bash"]
