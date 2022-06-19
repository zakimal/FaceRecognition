from email import header
import logging
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import face_recognition

import azure.functions as func

def draw_faces(img, locs):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img, mode="RGBA")
    for top, right, bottom, left in locs:
        draw.rectangle((left, top, right, bottom), outline="lime", width=2)
    return img

def img_to_base64(image):
    im_file = BytesIO()
    image.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()
    return base64.b64encode(im_bytes)

def base64_to_image(base64_bytes):
    im_bytes = base64.b64decode(base64_bytes)
    im_file = BytesIO(im_bytes)
    return Image.open(im_file)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    method = req.params.get('method')
    if not method:
        method = "hog"

    req_json = req.get_json()
    b64_bytes = req_json["image"].encode("utf-8")
    sep = b64_bytes.decode("utf-8").find(",")
    b64_bytes = b64_bytes.decode("utf-8")[sep + 1:].encode("utf-8")
    im = base64_to_image(b64_bytes)
    im = im.convert("RGB")
    im = np.array(im)
    locs = face_recognition.face_locations(im, number_of_times_to_upsample=1, model=method)
    facial_landmarks = face_recognition.face_landmarks(im, locs)

    pil_image = Image.fromarray(im)
    for face_landmarks in facial_landmarks:
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

        # Sparkle the eyes
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

    return func.HttpResponse(json.dumps({"image": "data:image/jpeg;base64," + img_to_base64(pil_image).decode("utf-8")}), headers={"Access-Control-Allow-Origin": "*"})
