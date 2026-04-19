# ============================================================
# AWS Glue ETL Job
# Proyecto: Churn de clientes en telecomunicaciones
# Objetivo:
#   1. Leer datos crudos desde el catálogo de Glue
#   2. Limpiar espacios en blanco en columnas clave
#   3. Normalizar vacíos a null
#   4. Convertir TotalCharges a tipo numérico
#   5. Crear la columna derivada tenure_group
#   6. Escribir la salida en formato Parquet en S3
# ============================================================

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

# Funciones de PySpark usadas en la transformación
from pyspark.sql.functions import col, trim, when, coalesce, lit

# ------------------------------------------------------------
# 1. Inicialización del job de Glue
# ------------------------------------------------------------
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ------------------------------------------------------------
# 2. Lectura de datos desde Glue Data Catalog
#    Ajusta database y table con tus nombres reales
# ------------------------------------------------------------
source_dyf = glueContext.create_dynamic_frame.from_catalog(
    database="db_telco_churn",
    table_name="raw_telco_customer_churn"
)

# Convertimos a DataFrame de Spark para aplicar transformaciones
df = source_dyf.toDF()

# ------------------------------------------------------------
# 3. Limpieza de espacios en blanco
#    Se aplica trim a columnas de texto y a TotalCharges,
#    ya que esa columna suele venir como string con espacios
# ------------------------------------------------------------
columns_to_clean = [
    "gender",
    "Partner",
    "Dependents",
    "Contract",
    "PaymentMethod",
    "TotalCharges"
]

from pyspark.sql.types import StringType

for c in df.columns:
    if isinstance(df.schema[c].dataType, StringType):
        df = df.withColumn(c, trim(col(c)))

df = df.withColumn("TotalCharges", col("TotalCharges").cast("string"))
df = df.withColumn(
    "TotalCharges",
    when(col("TotalCharges") == "", None).otherwise(col("TotalCharges"))
)

# ------------------------------------------------------------
# 4. Normalización de strings vacíos a null
#    Esto evita problemas posteriores en análisis y cast
# ------------------------------------------------------------
for c in columns_to_clean:
    df = df.withColumn(
        c,
        when(col(c) == "", None).otherwise(col(c))
    )

# ------------------------------------------------------------
# 5. Conversión de tipos
#    tenure debe ser entero
#    TotalCharges debe ser numérico para análisis y ML
# ------------------------------------------------------------
df = df.withColumn("tenure", col("tenure").cast("int"))
df = df.withColumn("TotalCharges", col("TotalCharges").cast("double"))

# Opcional:
# Reemplazar null en TotalCharges por 0.0
# Esto ayuda si más adelante usarás el dataset para ML
df = df.withColumn(
    "TotalCharges",
    coalesce(col("TotalCharges"), lit(0.0))
)

# ------------------------------------------------------------
# 6. Creación de variable derivada tenure_group
#    Segmenta la antigüedad del cliente para análisis de churn
# ------------------------------------------------------------
df = df.withColumn(
    "tenure_group",
    when(col("tenure") <= 12, "0-1yr")
    .when((col("tenure") > 12) & (col("tenure") <= 24), "1-2yr")
    .when((col("tenure") > 24) & (col("tenure") <= 48), "2-4yr")
    .when((col("tenure") > 48) & (col("tenure") <= 60), "4-5yr")
    .otherwise("5+yr")
)

# ------------------------------------------------------------
# 7. Validaciones básicas
#    Estas líneas ayudan durante pruebas; puedes dejarlas
#    mientras desarrollas y quitarlas si quieres después
# ------------------------------------------------------------
df.printSchema()
df.select(
    "gender",
    "Partner",
    "Dependents",
    "Contract",
    "PaymentMethod",
    "TotalCharges",
    "tenure",
    "tenure_group"
).show(10, truncate=False)

# ------------------------------------------------------------
# 8. Convertir de nuevo a DynamicFrame
#    Glue escribe normalmente desde DynamicFrame
# ------------------------------------------------------------
target_dyf = DynamicFrame.fromDF(df, glueContext, "target_dyf")

# ------------------------------------------------------------
# 9. Escritura en S3 en formato Parquet
#    Ajusta la ruta con tu bucket real
# ------------------------------------------------------------
glueContext.write_dynamic_frame.from_options(
    frame=target_dyf,
    connection_type="s3",
    connection_options={
        "path": "s3://aws-churn-cesar-2026/processed/"
    },
    format="parquet"
)

# ------------------------------------------------------------
# 10. Cierre del job
# ------------------------------------------------------------
job.commit()