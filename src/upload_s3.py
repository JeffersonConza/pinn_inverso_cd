# src/upload_s3.py
import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv

from src.config import Config

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET", "pinns-models-repository")
MODEL_PATH = os.getenv("MODEL_PATH", Config.MODEL_PATH)
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "models/inverse_drag_pinn.pt")


def upload_model_to_s3(
    local_path: str = MODEL_PATH,
    bucket_name: str = S3_BUCKET,
    s3_key: str = S3_MODEL_KEY,
) -> bool:
    """Sube los pesos entrenados del modelo PINN inverso a un bucket de Amazon S3.

    Args:
        local_path (str): Ruta local del archivo de pesos .pt
        bucket_name (str): Nombre del bucket S3 de destino
        s3_key (str): Prefijo y nombre de archivo dentro del bucket

    Returns:
        bool: True si la subida fue exitosa, False en caso contrario.
    """
    if not os.path.exists(local_path):
        print(f"❌ Error: El archivo local '{local_path}' no existe. Ejecuta primero 'python -m src.train'.")
        return False

    print(f"📦 Conectando con Amazon S3...")
    print(f"   Bucket: {bucket_name}")
    print(f"   Clave de destino: {s3_key}")
    print(f"   Archivo local: {local_path} ({os.path.getsize(local_path) / 1024:.2f} KB)")

    try:
        s3_client = boto3.client("s3")
        s3_client.upload_file(local_path, bucket_name, s3_key)
        print(f"✅ ¡Éxito! Modelo subido a s3://{bucket_name}/{s3_key}")
        return True
    except NoCredentialsError:
        print("⚠️ Advertencia: No se encontraron credenciales de AWS (ej. ~/.aws/credentials o AWS_ACCESS_KEY_ID).")
        return False
    except ClientError as e:
        print(f"❌ Error de cliente AWS S3: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False


if __name__ == "__main__":
    upload_model_to_s3()
