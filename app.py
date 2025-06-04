from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile, json, io
from processor import load_categories, run_processing, load_json
from pycocotools.coco import COCO
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)

@app.route('/get-categories', methods=['POST'])
def get_categories():
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        file.save(tmp.name)
        categories = load_categories(tmp.name)
        return jsonify(categories)

@app.route('/process', methods=['POST'])
def process():
    file = request.files['file']
    categories = request.form.getlist('categories[]')
    url_map = json.loads(request.form.get("filename_to_url"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        file.save(tmp.name)
        result = run_processing(tmp.name, categories, url_map)

        buffer = io.BytesIO()
        buffer.write(json.dumps(result['output'], indent=2).encode())
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="converted_annotations.json")

# Added in V2
@app.route('/preview-masks', methods=['POST'])
def preview_masks():
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        file.save(tmp.name)
        coco_data = load_json(tmp.name)
        coco = COCO(tmp.name)

        id_to_cat = {cat['id']: cat['name'] for cat in coco_data['categories']}
        results = {}

        for img in coco_data['images']:
            image_id = img['id']
            ann_ids = coco.getAnnIds(imgIds=image_id)
            masks = []
            for ann_id in ann_ids:
                ann = coco.loadAnns(ann_id)[0]
                mask = coco.annToMask(ann)
                mask = (mask * 255).astype(np.uint8)
                _, buffer = cv2.imencode('.png', mask)
                b64 = base64.b64encode(buffer).decode()
                masks.append(f"data:image/png;base64,{b64}")
            results[img['file_name']] = masks
        return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)