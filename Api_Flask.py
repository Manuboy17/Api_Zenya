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

# ✅ CARGAR MODELO
print("🔄 Cargando modelo ONNX...")
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 2
session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

MODEL = ort.InferenceSession('best.onnx', session_options)
print("✅ Modelo cargado")

with open('clases.txt', 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

print(f"📚 {len(CLASSES)} clases cargadas")

def preprocess_image(image_base64):
    """
    Preprocesamiento NORMALIZADO (0-1)
    La mayoría de modelos YOLO se entrenan así
    """
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        image_np = np.array(image)
        
        # Redimensionar
        image_resized = cv2.resize(image_np, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # ⚡ NORMALIZAR A [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Transponer
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_transposed, axis=0)
        
        print(f"📐 Normalizado [0-1]: Shape {image_batch.shape}, Range [{image_batch.min():.3f}-{image_batch.max():.3f}]")
        
        return image_batch
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


def apply_nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression"""
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
    return jsonify({
        "status": "online",
        "message": "API Lengua de Señas",
        "version": "3.0",
        "classes": len(CLASSES)
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/test-predict')
def test_predict():
    return jsonify({
        "model_loaded": True,
        "classes": len(CLASSES),
        "classes_sample": CLASSES[:10],
        "input_shape": str(MODEL.get_inputs()[0].shape),
        "output_shape": str(MODEL.get_outputs()[0].shape)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        confidence_threshold = data.get('confidence', 0.05)  # MUY BAJO
        iou_threshold = data.get('iou_threshold', 0.45)
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No image'}), 400
        
        print(f"\n{'='*60}")
        print(f"📥 Threshold: {confidence_threshold}")
        
        # Preprocesar
        input_tensor = preprocess_image(image_base64)
        
        # Inferencia
        outputs = MODEL.run(None, {MODEL.get_inputs()[0].name: input_tensor})
        predictions = outputs[0][0]  # Shape: (85, 8400) para YOLOv5/v8
        
        print(f"📊 Output: {predictions.shape}")
        
        # Analizar confianzas
        confidences = predictions[4, :]  # Fila de objectness
        max_conf = confidences.max()
        mean_conf = confidences.mean()
        
        print(f"📈 Max conf: {max_conf:.6f}")
        print(f"📈 Mean conf: {mean_conf:.6f}")
        print(f"📈 > 0.01: {(confidences > 0.01).sum()}")
        print(f"📈 > 0.05: {(confidences > 0.05).sum()}")
        print(f"📈 > 0.1: {(confidences > 0.1).sum()}")
        print(f"📈 > 0.3: {(confidences > 0.3).sum()}")
        
        # Post-procesamiento
        boxes = []
        scores = []
        class_ids = []
        raw_count = 0
        
        # Iterar sobre detecciones (8400 anchors)
        for i in range(predictions.shape[1]):
            objectness = predictions[4, i]  # Confianza de objeto
            
            if objectness > confidence_threshold:
                raw_count += 1
                
                # Clases (85 - 5 = 80 clases para COCO, o 27 para tu modelo)
                class_scores = predictions[5:, i]
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                
                # Confianza final = objectness * class_confidence
                final_confidence = objectness * class_conf
                
                if raw_count <= 3:  # Log solo primeras 3
                    class_name = CLASSES[class_id] if class_id < len(CLASSES) else f'class_{class_id}'
                    print(f"  🔸 #{raw_count}: obj={objectness:.4f}, cls={class_conf:.4f}, final={final_confidence:.4f}, class={class_name}")
                
                if final_confidence > confidence_threshold:
                    # Coordenadas YOLO: [x_center, y_center, width, height]
                    x_center = predictions[0, i]
                    y_center = predictions[1, i]
                    width = predictions[2, i]
                    height = predictions[3, i]
                    
                    # Convertir a [x1, y1, x2, y2]
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(final_confidence))
                    class_ids.append(int(class_id))
        
        print(f"🔍 Raw: {raw_count}, Filtered: {len(boxes)}")
        
        # NMS
        if len(boxes) > 0:
            boxes_array = np.array(boxes)
            scores_array = np.array(scores)
            keep = apply_nms(boxes_array, scores_array, iou_threshold)
            
            final_boxes = boxes_array[keep].tolist()
            final_scores = [scores_array[i] for i in keep]
            final_class_ids = [class_ids[i] for i in keep]
            final_letters = [CLASSES[cid] if cid < len(CLASSES) else f'class_{cid}' for cid in final_class_ids]
            
            print(f"✅ Final: {len(keep)}, Letters: {final_letters}")
            print(f"{'='*60}\n")
            
            return jsonify({
                'success': True,
                'detections': {
                    'num_detections': len(keep),
                    'boxes': final_boxes,
                    'scores': final_scores,
                    'class_ids': final_class_ids,
                    'letters': final_letters
                }
            })
        
        print(f"❌ No detections")
        print(f"{'='*60}\n")
        
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
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
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
    print(f"\n🚀 Servidor en puerto {port}")
    print(f"📚 {len(CLASSES)} clases\n")
    app.run(host='0.0.0.0', port=port, debug=False)
