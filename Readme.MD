# Pyspark vector data ingestion (via Docker)
## Prerequisites
### Create the docker network
docker network create alloydb-spark-net
### Run the AlloyDB Omni via Docker
docker run  --rm --name my-omni --network alloydb-spark-net \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  -d google/alloydbomni:latest
### Enable Vector storage type
CREATE EXTENSION IF NOT EXISTS vector;
### Create the table
CREATE TABLE IF NOT EXISTS pdf_documents (
    id SERIAL PRIMARY KEY,
    file_name TEXT,
    content TEXT,
    embedding vector(1536) -- Must match OpenAI text-embedding-3-small
);
### Build the image
docker build -t alloydb-omni-vector-pyspark-ingestion-app .

### Run the container
docker run --env-file ./.env --network alloydb-spark-net --rm alloydb-omni-vector-pyspark-ingestion-app