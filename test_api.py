import requests
import base64
import os

# Verificar que el archivo existe
image_path = 'test.jpg'
if not os.path.exists(image_path):
    print(f"Error: No se encontró la imagen '{image_path}'")
    print(f"Busca una imagen en: {os.getcwd()}")
    exit(1)

print(f"Leyendo imagen: {image_path}")

# Leer y codificar imagen
with open(image_path, 'rb') as f:
    image_bytes = f.read()
    print(f"Tamaño del archivo: {len(image_bytes)} bytes")
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    print(f"Tamaño base64: {len(image_b64)} caracteres")

# Enviar a la API
print("Enviando petición a la API...")
response = requests.post('http://127.0.0.1:5000/predict', 
                        json={'image': image_b64})

print("Status Code:", response.status_code)
print("Response:", response.json())
