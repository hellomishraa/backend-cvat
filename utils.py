import base64
import cv2
import numpy as np
import requests

def array_encode(np_array):
    success, encoded_img = cv2.imencode('.png', np_array)
    if not success:
        raise ValueError("Could not encode image")
    return base64.b64encode(encoded_img).decode('utf-8')

def get_image_dimensions_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return (img.shape[1], img.shape[0]) if img is not None else None
    except:
        return None

def get_mask_center(mask):
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return {"x": cx, "y": cy}