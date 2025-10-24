from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import cv2
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Cargar modelo UNA SOLA VEZ
print("🔄 Cargando modelo...")
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 2
session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

MODEL = ort.InferenceSession('best.onnx', session_options)
print("✅ Modelo cargado")

with open('clases.txt', 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

def preprocess_image(image_base64):
    """Preprocesamiento ORIGINAL"""
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (640, 640))
    
    # ⚡ SOLUCIÓN: Convertir a float32 ANTES de dividir
    image_float = image_resized.astype(np.float32)
    image_normalized = image_float / 255.0
    
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_batch = np.expand_dims(image_transposed, axis=0)
    return image_batch


def apply_nms(boxes, scores, iou_threshold=0.45):
    """NMS Original"""
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

@app.route('/')
def home():
    return jsonify({"status": "online", "message": "API funcionando"})

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        confidence_threshold = data.get('confidence', 0.5)
        iou_threshold = data.get('iou_threshold', 0.45)
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No image'}), 400
        
        # Preprocesar
        input_tensor = preprocess_image(image_base64)
        
        # Inferencia
        outputs = MODEL.run(None, {MODEL.get_inputs()[0].name: input_tensor})
        predictions = outputs[0][0]
        
        # Post-procesamiento ORIGINAL
        boxes = []
        scores = []
        class_ids = []
        
        for detection in predictions.T:
            confidence = detection[4]
            if confidence > confidence_threshold:
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                if class_confidence > confidence_threshold:
                    x_center, y_center, width, height = detection[0:4]
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(class_confidence))
                    class_ids.append(int(class_id))
        
        # NMS
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            keep_indices = apply_nms(boxes, scores, iou_threshold)
            
            final_boxes = boxes[keep_indices].tolist()
            final_scores = [scores[i] for i in keep_indices]
            final_class_ids = [class_ids[i] for i in keep_indices]
            final_letters = [CLASSES[i] for i in final_class_ids]
            
            return jsonify({
                'success': True,
                'detections': {
                    'num_detections': len(keep_indices),
                    'boxes': final_boxes,
                    'scores': final_scores,
                    'class_ids': final_class_ids,
                    'letters': final_letters
                }
            })
        
        return jsonify({
            'success': True,
            'detections': {
                'num_detections': 0,
                'boxes': [],
                'scores': [],
                'class_ids': [],
                'letters': []
            }
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
