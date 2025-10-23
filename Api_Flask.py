from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import base64
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cargar nombres de clases (letras)
try:
    with open('clases.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✓ Clases cargadas: {class_names}")
except:
    class_names = []
    print("⚠ No se encontró classes.txt")

# Cargar modelo ONNX
try:
    session = ort.InferenceSession('best.onnx')
    print("✓ Modelo ONNX cargado correctamente")
    input_shape = session.get_inputs()[0].shape
    print(f"Forma de entrada esperada: {input_shape}")
except Exception as e:
    print(f"✗ Error al cargar el modelo: {e}")
    session = None

def preprocess_image(image_data, input_size=640):
    """Preprocesa la imagen para YOLO usando PIL"""
    try:
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        original_size = img.size
        img = img.resize((input_size, input_size))
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, original_size
    except Exception as e:
        raise ValueError(f"Error al preprocesar imagen: {str(e)}")

def calculate_iou(box1, box2):
    """Calcula Intersection over Union entre dos cajas"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convertir de (center_x, center_y, width, height) a (x1, y1, x2, y2)
    box1_x1 = x1 - w1/2
    box1_y1 = y1 - h1/2
    box1_x2 = x1 + w1/2
    box1_y2 = y1 + h1/2
    
    box2_x1 = x2 - w2/2
    box2_y1 = y2 - h2/2
    box2_x2 = x2 + w2/2
    box2_y2 = y2 + h2/2
    
    # Calcular intersección
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calcular unión
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area

def non_max_suppression(boxes, scores, class_ids, iou_threshold=0.45):
    """Aplica Non-Maximum Suppression para eliminar detecciones duplicadas"""
    if len(boxes) == 0:
        return [], [], []
    
    # Ordenar por score descendente
    indices = np.argsort(scores)[::-1]
    
    keep_boxes = []
    keep_scores = []
    keep_class_ids = []
    
    while len(indices) > 0:
        # Mantener la caja con mayor score
        current = indices[0]
        keep_boxes.append(boxes[current])
        keep_scores.append(scores[current])
        keep_class_ids.append(class_ids[current])
        
        if len(indices) == 1:
            break
        
        # Calcular IoU con el resto de cajas
        rest_indices = indices[1:]
        ious = [calculate_iou(boxes[current], boxes[i]) for i in rest_indices]
        
        # Mantener solo las cajas con IoU bajo (no superpuestas)
        indices = rest_indices[np.array(ious) < iou_threshold]
    
    return keep_boxes, keep_scores, keep_class_ids

def postprocess_predictions(output, conf_threshold=0.5, iou_threshold=0.45, target_classes=None):
    """Post-procesa las predicciones de YOLO con NMS"""
    predictions = output[0]
    predictions = predictions.transpose()
    
    boxes = []
    scores = []
    class_ids = []
    
    # Primera fase: filtrar por confianza y clase
    for pred in predictions:
        x, y, w, h = pred[:4]
        class_scores = pred[4:]
        
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if target_classes is not None:
            if class_id not in target_classes:
                continue
        
        if confidence > conf_threshold:
            boxes.append([float(x), float(y), float(w), float(h)])
            scores.append(float(confidence))
            class_ids.append(int(class_id))
    
    # Segunda fase: aplicar NMS
    if len(boxes) > 0:
        boxes, scores, class_ids = non_max_suppression(
            boxes, scores, class_ids, iou_threshold
        )
    
    # Convertir class_ids a letras
    letters = []
    for class_id in class_ids:
        if class_id < len(class_names):
            letters.append(class_names[class_id])
        else:
            letters.append(f"clase_{class_id}")
    
    return {
        'num_detections': len(boxes),
        'boxes': boxes,
        'scores': scores,
        'class_ids': class_ids,
        'letters': letters
    }

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({'error': 'No se encontró el campo "image"'}), 400
        
        target_classes = data.get('classes', None)
        conf_threshold = data.get('confidence', 0.5)
        iou_threshold = data.get('iou_threshold', 0.45)
        
        input_data, original_size = preprocess_image(data['image'])
        
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        
        results = postprocess_predictions(
            outputs[0], 
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            target_classes=target_classes
        )
        
        return jsonify({
            'success': True,
            'detections': results,
            'original_image_size': original_size,
            'message': f'Letras detectadas: {", ".join(results["letters"])}'
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'API de reconocimiento de lengua de señas',
        'clases_disponibles': class_names,
        'total_clases': len(class_names),
        'endpoint': '/predict',
        'method': 'POST'
    })

if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)
