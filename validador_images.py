from PIL import Image

try:
    img = Image.open('test.jpg')
    print(f"✓ Imagen válida: {img.size}, formato: {img.format}")
except Exception as e:
    print(f"✗ Imagen inválida: {e}")