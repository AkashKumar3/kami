from PIL import Image
import numpy as np
from transformers import AutoModel
import torch

Image.MAX_IMAGE_PIXELS = None

def split_tall_image(img, chunk_height=9000):
    width, height = img.size
    for y in range(0, height, chunk_height):
        upper = y
        lower = min(y + chunk_height, height)
        yield img.crop((0, upper, width, lower))

def read_image_as_np_array(image):
    return np.array(image.convert("L").convert("RGB"))

model = AutoModel.from_pretrained(
    "ragavsachdeva/magi", trust_remote_code=True
).cpu().eval()

def ocr_image(img):
    img_np = read_image_as_np_array(img)
    with torch.no_grad():
        results = model.predict_detections_and_associations([img_np])
        text_bboxes = [res["texts"] for res in results]
        ocr_results = model.predict_ocr([img_np], text_bboxes)
    return "\n".join(ocr_results[0])
