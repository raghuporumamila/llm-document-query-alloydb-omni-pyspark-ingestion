import numpy as np
import pandas as pd
import tiktoken
from openai import OpenAI
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, lit, concat, concat_ws
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
from pypdf import PdfReader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("AlloyDB_PDF_Ingestion") \
    .config("spark.jars", "./postgresql-42.7.7.jar") \
    .getOrCreate()

# 2. Local PDF Extraction
def extract_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
    except:
        return None

pdf_folder = "./docs"
pdf_data = [(f, extract_text(os.path.join(pdf_folder, f)))
            for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

schema = StructType([
    StructField("file_name", StringType(), True),
    StructField("content", StringType(), True)
])
df = spark.createDataFrame([d for d in pdf_data if d[1]], schema)

# 3. Embedding Logic (with chunking support)
@pandas_udf(ArrayType(FloatType()))
def get_embeddings(texts: pd.Series) -> pd.Series:
    client = OpenAI(api_key=api_key)
    model = "text-embedding-3-small"
    encoding = tiktoken.encoding_for_model(model)
    max_tokens = 8000

    def get_safe_embedding(text):
        if not text or not isinstance(text, str):
            return [0.0] * 1536

        tokens = encoding.encode(text)

        if len(tokens) <= max_tokens:
            res = client.embeddings.create(input=[text], model=model)
            return res.data[0].embedding

        chunks = [tokens[i: i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        chunk_embeddings = []

        for chunk in chunks:
            res = client.embeddings.create(input=[chunk], model=model)
            chunk_embeddings.append(res.data[0].embedding)

        avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
        return avg_embedding

    return texts.apply(get_safe_embedding)

# 4. Transform: Format for the 'embedding' column in the DB
# We convert the array to a string format "[v1,v2...]" so pgvector can cast it
df_final = df.withColumn("raw_embedding", get_embeddings(col("content"))) \
             .withColumn("embedding",
                         concat(lit("["), concat_ws(",", col("raw_embedding")), lit("]")))

# 5. JDBC Write to AlloyDB Omni
jdbc_url = "jdbc:postgresql://my-omni:5432/postgres"
db_props = {
    "user": "postgres",
    "password": "password",
    "driver": "org.postgresql.Driver",
    "stringtype": "unspecified" # This allows the string to be cast to 'vector' automatically
}

# We select only the columns defined in your table (id is SERIAL, so it populates automatically)
df_final.select("file_name", "content", "embedding") \
        .write.jdbc(url=jdbc_url, table="pdf_documents", mode="append", properties=db_props)

print("Ingestion successful!")