# Title: nb_sm_telco_churn

# Set environment variables for sagemaker_studio imports

import os
os.environ['DataZoneProjectId'] = 'b6vrhssjkxkyhz'
os.environ['DataZoneDomainId'] = 'dzd-3ogpnc7m7un0iv'
os.environ['DataZoneEnvironmentId'] = 'bm8fuehgciatd3'
os.environ['DataZoneDomainRegion'] = 'us-east-1'

# create both a function and variable for metadata access
_resource_metadata = None

def _get_resource_metadata():
    global _resource_metadata
    if _resource_metadata is None:
        _resource_metadata = {
            "AdditionalMetadata": {
                "DataZoneProjectId": "b6vrhssjkxkyhz",
                "DataZoneDomainId": "dzd-3ogpnc7m7un0iv",
                "DataZoneEnvironmentId": "bm8fuehgciatd3",
                "DataZoneDomainRegion": "us-east-1",
            }
        }
    return _resource_metadata
metadata = _get_resource_metadata()

"""
Logging Configuration

Purpose:
--------
This sets up the logging framework for code executed in the user namespace.
"""

from typing import Optional


def _set_logging(log_dir: str, log_file: str, log_name: Optional[str] = None):
    import os
    import logging
    from logging.handlers import RotatingFileHandler

    level = logging.INFO
    max_bytes = 5 * 1024 * 1024
    backup_count = 5

    # fallback to /tmp dir on access, helpful for local dev setup
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = "/tmp/kernels/"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger() if not log_name else logging.getLogger(log_name)
    logger.handlers = []
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Rotating file handler
    fh = RotatingFileHandler(filename=log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging initialized for {log_name}.")


_set_logging("/var/log/computeEnvironments/kernel/", "kernel.log")
_set_logging("/var/log/studio/data-notebook-kernel-server/", "metrics.log", "metrics")

import logging
from sagemaker_studio import ClientConfig, sqlutils, sparkutils, dataframeutils

logger = logging.getLogger(__name__)
logger.info("Initializing sparkutils")
spark = sparkutils.init()
logger.info("Finished initializing sparkutils")

def _reset_os_path():
    """
    Reset the process's working directory to handle mount timing issues.
    
    This function resolves a race condition where the Python process starts
    before the filesystem mount is complete, causing the process to reference
    old mount paths and inodes. By explicitly changing to the mounted directory
    (/home/sagemaker-user), we ensure the process uses the correct, up-to-date
    mount point.
    
    The function logs stat information (device ID and inode) before and after
    the directory change to verify that the working directory is properly
    updated to reference the new mount.
    
    Note:
        This is executed at module import time to ensure the fix is applied
        as early as possible in the kernel initialization process.
    """
    try:
        import os
        import logging

        logger = logging.getLogger(__name__)
        logger.info("---------Before------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)

        os.chdir("/home/sagemaker-user")

        logger.info("---------After------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)
    except Exception as e:
        logger.exception(f"Failed to reset working directory: {e}")

_reset_os_path()

# ------------------------------------------------------------
# 1. Importación de librerías
# ------------------------------------------------------------
import pandas as pd

# ------------------------------------------------------------
# 2. Definición de parámetros básicos
#    Ajusta el nombre del bucket según tu caso real
# ------------------------------------------------------------
# Ruta al archivo procesado (ajusta según tu bucket)
s3_path = "s3://aws-churn-cesar-2026/processed/"

# ------------------------------------------------------------
# 3. Lectura del dataset procesado desde S3
#    Se asume que el ETL ya dejó la salida en formato Parquet
# ------------------------------------------------------------

df = pd.read_parquet(s3_path)

df.head()

# ------------------------------------------------------------
# 4. Preparación de la variable objetivo
#    Convertimos Churn de texto a formato binario:
#    Yes -> 1
#    No  -> 0
# ------------------------------------------------------------
df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

# ------------------------------------------------------------
# 5. Eliminación de columnas no relevantes para el modelo
#    customerID es un identificador único y no aporta valor
#    predictivo al entrenamiento
# ------------------------------------------------------------
df = df.drop(columns=['customerid'])

# ------------------------------------------------------------
# 6. Conversión de variables categóricas a numéricas
#    get_dummies aplica one-hot encoding sobre columnas de texto
#    Esto es necesario porque XGBoost trabaja con valores numéricos
# ------------------------------------------------------------
df = pd.get_dummies(df)

# ------------------------------------------------------------
# 7. Separación de variables predictoras (X) y objetivo (y)
# ------------------------------------------------------------
X = df.drop('churn', axis=1)
y = df['churn']

# ------------------------------------------------------------
# 8. División del dataset en entrenamiento y prueba
#    Se utiliza 80% para train y 20% para test
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

import pandas as pd
# ------------------------------------------------------------
# 9. Reorganización para SageMaker XGBoost
#    La variable objetivo debe quedar en la primera columna
# ------------------------------------------------------------
train_data = pd.concat([y_train, X_train], axis=1)
test_data = pd.concat([y_test, X_test], axis=1)

# ------------------------------------------------------------
# 10. Exportación de archivos CSV
#     - sin índice
#     - sin encabezado
#     Esto facilita el entrenamiento en SageMaker XGBoost
# ------------------------------------------------------------

# Convert boolean columns to integers (True -> 1, False -> 0)
train_data = train_data.astype({col: int for col in train_data.select_dtypes(include='bool').columns})
test_data = test_data.astype({col: int for col in test_data.select_dtypes(include='bool').columns})

train_data.to_csv('train.csv', index=False, header=False)
test_data.to_csv('test.csv', index=False, header=False)

# ------------------------------------------------------------
# 11. Subida de archivos a S3
#     Estos archivos serán usados por SageMaker como input
# ------------------------------------------------------------
import boto3

bucket = "aws-churn-cesar-2026"

s3 = boto3.client('s3')

s3.upload_file('train.csv', bucket, 'ml/train/train.csv')
s3.upload_file('test.csv', bucket, 'ml/test/test.csv')

import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

# ------------------------------------------------------------
# 12. Inicialización de sesión de SageMaker
# ------------------------------------------------------------
sess = sagemaker.Session()
region = sess.boto_region_name
role = sagemaker.get_execution_role()

# ------------------------------------------------------------
# 13. Selección de la imagen del algoritmo XGBoost
# ------------------------------------------------------------
container = image_uris.retrieve("xgboost", region, version="1.2-1")

# ------------------------------------------------------------
# 14. IDs reales de la VPC creados previamente
# Reemplazar por tus valores reales
# ------------------------------------------------------------
private_subnets = [
    "subnet-0d9853ae6c60da87c",  # subnet-private-1
    "subnet-0ba84f45180a0485f"   # subnet-private-2
]

security_group_ids = [
    "sg-0b6b55675858c15ed"       # sg-sagemaker-ml-private
]

# ------------------------------------------------------------
# 15. Configuración del estimador
#     - instance_count: número de instancias
#     - instance_type: tipo de máquina para entrenamiento
#     - output_path: ruta donde se guardará el artefacto del modelo
# ------------------------------------------------------------
estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://aws-churn-cesar-2026/ml/output',
    sagemaker_session=sess,
    subnets=private_subnets,
    security_group_ids=security_group_ids
)

# ------------------------------------------------------------
# 16. Definición de hiperparámetros básicos
#     - objective: clasificación binaria
#     - num_round: número de iteraciones del entrenamiento
# ------------------------------------------------------------
estimator.set_hyperparameters(
    objective="binary:logistic",
    num_round=100
)

# ------------------------------------------------------------
# 16. Entrenamiento del modelo
#     Se definen los canales de entrada:
#     - train
#     - validation
# ------------------------------------------------------------
train_input = TrainingInput(
    's3://aws-churn-cesar-2026/ml/train/train.csv',
    content_type='text/csv'
)
validation_input = TrainingInput(
    's3://aws-churn-cesar-2026/ml/test/test.csv',
    content_type='text/csv'
)

estimator.fit({
    'train': train_input,
    'validation': validation_input
})

print("Entrenamiento lanzado correctamente en SageMaker.")

# ============================================================
# 17. EVALUACIÓN DEL MODELO EN EL NOTEBOOK
# ------------------------------------------------------------
# Objetivo:
#   - Entrenar un modelo equivalente en el notebook usando
#     el mismo algoritmo base (XGBoost)
#   - Obtener probabilidades de churn sobre el conjunto de prueba
#   - Calcular métricas de desempeño
#   - Generar evidencias visuales:
#       * Curva ROC
#       * AUC
#       * Matriz de confusión
#
# Nota:
#   Este bloque NO reemplaza el entrenamiento en SageMaker.
#   El entrenamiento principal ya se ejecutó en AWS.
#   Este bloque se utiliza para facilitar la evaluación y
#   visualización de resultados dentro del notebook.
# ============================================================

# ------------------------------------------------------------
# 17.1 Importación de librerías para evaluación
# ------------------------------------------------------------
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ------------------------------------------------------------
# 17.2 Entrenamiento de un modelo equivalente en el notebook
# ------------------------------------------------------------
# Se utiliza XGBClassifier porque permite entrenar el mismo
# tipo de algoritmo (XGBoost) pero con una interfaz sencilla
# para generar probabilidades y métricas localmente.
#
# Este modelo se entrena con los mismos datos X_train / y_train
# ya preparados previamente en el notebook.
# ------------------------------------------------------------
eval_model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# Entrenamiento del modelo en el notebook
eval_model.fit(X_train, y_train)

print("Modelo de evaluación entrenado correctamente en el notebook.")

# ------------------------------------------------------------
# 17.3 Generación de probabilidades y predicciones
# ------------------------------------------------------------
# predict_proba devuelve la probabilidad para cada clase.
# La segunda columna [:, 1] corresponde a la probabilidad
# estimada de churn = 1.
# ------------------------------------------------------------
y_proba = eval_model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# Se convierten las probabilidades en predicciones binarias
# usando un umbral de 0.5
# ------------------------------------------------------------
y_pred = (y_proba >= 0.5).astype(int)

print("Probabilidades y predicciones generadas correctamente.")

# ------------------------------------------------------------
# 17.4 Cálculo del AUC
# ------------------------------------------------------------
# El AUC (Area Under the Curve) resume la capacidad del modelo
# para discriminar entre clientes con churn y sin churn.
# Valores más cercanos a 1 indican mejor desempeño.
# ------------------------------------------------------------
auc_score = roc_auc_score(y_test, y_proba)
print(f"AUC del modelo: {auc_score:.4f}")

# ------------------------------------------------------------
# 17.5 Cálculo de la curva ROC
# ------------------------------------------------------------
# fpr = False Positive Rate
# tpr = True Positive Rate
# thresholds = umbrales evaluados por la curva
# ------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# ------------------------------------------------------------
# 17.6 Visualización de la curva ROC
# ------------------------------------------------------------
# La línea diagonal representa el desempeño aleatorio.
# Mientras más se acerque la curva a la esquina superior
# izquierda, mejor es la capacidad de discriminación.
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Churn Prediction Model")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 17.7 Cálculo de la matriz de confusión
# ------------------------------------------------------------
# La matriz de confusión resume:
#   - Verdaderos negativos (TN)
#   - Falsos positivos (FP)
#   - Falsos negativos (FN)
#   - Verdaderos positivos (TP)
# ------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

print("Matriz de confusión:")
print(cm)

# ------------------------------------------------------------
# 17.8 Visualización de la matriz de confusión
# ------------------------------------------------------------
# display_labels:
#   0 = No Churn
#   1 = Churn
# ------------------------------------------------------------
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("Confusion Matrix - Churn Prediction Model")
plt.show()

# ------------------------------------------------------------
# 17.9 Reporte de clasificación
# ------------------------------------------------------------
# El reporte muestra métricas como:
#   - precision
#   - recall
#   - f1-score
#   - support
#
# Esto ayuda a interpretar mejor el desempeño del modelo
# en cada clase.
# ------------------------------------------------------------
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))