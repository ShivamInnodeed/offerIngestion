# Build context must be the parent directory (d:\test\ingestion\)
# Build with: docker build -f offerIngestion/Dockerfile -t offer-ingestion-api .

FROM python:3.11-slim-bookworm

RUN set -eux; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends build-essential; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY offerIngestion/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --default-timeout=1200 --retries 20 -r /tmp/requirements.txt

# Optional offline wheels (for air-gapped servers). Safe to keep empty.
COPY offerIngestion/wheels/ /tmp/wheels/
RUN set -e; \
    if [ -d /tmp/wheels ] && [ "$(ls -A /tmp/wheels 2>/dev/null | wc -l)" -gt 0 ]; then \
      if ls /tmp/wheels/*.whl >/dev/null 2>&1; then pip install --no-cache-dir /tmp/wheels/*.whl; fi; \
      if ls /tmp/wheels/*.zip >/dev/null 2>&1; then pip install --no-cache-dir /tmp/wheels/*.zip; fi; \
    fi

# Copy shared modules from ingestion/ (es_kb, embedding builders, etc.)
COPY ingestion/es_kb/                                      /app/ingestion/es_kb/
COPY ingestion/common/                                     /app/ingestion/common/
COPY ingestion/build_embedding_payloads.py                 /app/ingestion/
COPY ingestion/build_sentence_transformer_embeddings.py    /app/ingestion/
COPY ingestion/chunk_embedding_payloads.py                 /app/ingestion/

# Copy offerIngestion source
COPY offerIngestion/__init__.py             /app/offerIngestion/
COPY offerIngestion/offer_ingestion_flow.py /app/offerIngestion/
COPY offerIngestion/xml_parser.py           /app/offerIngestion/
COPY offerIngestion/offer_llm_enricher.py   /app/offerIngestion/
COPY offerIngestion/offer_llm_json_parse.py /app/offerIngestion/
COPY offerIngestion/offer_llm_system_prompt.py /app/offerIngestion/
COPY offerIngestion/offer_es_mapping.py     /app/offerIngestion/
COPY offerIngestion/offer_standalone_index.py /app/offerIngestion/
COPY offerIngestion/offer_config.json       /app/offerIngestion/
COPY offerIngestion/script.py               /app/offerIngestion/
COPY offerIngestion/bootstrap_oracle_schema.py /app/offerIngestion/
COPY offerIngestion/api/                    /app/offerIngestion/api/
COPY offerIngestion/common/                 /app/offerIngestion/common/

# XML data directory (will be mounted as volume in production)
COPY offerIngestion/all_xml_data/           /app/offerIngestion/all_xml_data/

ENV PYTHONPATH=/app/offerIngestion:/app/ingestion

# Pre-download embedding model into the image so the server does not need HuggingFace access.
RUN python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('all-MiniLM-L6-v2'); m.save('/app/models/all-MiniLM-L6-v2')"

# Persistent directories
RUN mkdir -p /app/data/output /app/data/logs

# Container environment defaults
ENV INGESTION_DIR=/app/ingestion \
    OFFER_CONFIG_PATH=/app/offerIngestion/offer_config.json \
    OFFER_OUTPUT_DIR=/app/data/output \
    OFFER_LOG_DIR=/app/data/logs \
    OFFER_XML_PATH=/app/offerIngestion/all_xml_data/all_offer_data1.xml \
    DEFAULT_OFFER_OUTPUT_DIR=/app/data/output \
    DEFAULT_OFFER_CONFIG_PATH=/app/offerIngestion/offer_config.json \
    LLM_MAX_RETRIES=3 \
    LLM_RETRY_DELAY_SECONDS=2

COPY offerIngestion/entrypoint.sh /app/offerIngestion/entrypoint.sh
RUN sed -i 's/\r$//' /app/offerIngestion/entrypoint.sh && chmod +x /app/offerIngestion/entrypoint.sh

EXPOSE 8000

WORKDIR /app/offerIngestion
CMD ["uvicorn", "api.scheduler_service:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
