import json
from pprint import pprint
import numpy as np
import base64
import cv2
import requests

def read_base64_image(uri:str)->np.ndarray:
   """ Converts bas64 string to image """

   split_uri = uri.split(',') 
   encoded_data = split_uri[0] if len(split_uri) == 0 else split_uri[0]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img


def image_to_base64(img: np.ndarray) -> str:
    """ Given a numpy 2D array, returns a JPEG image in base64 format """
    img_buffer = cv2.imencode('.jpg', img)[1]
    return base64.b64encode(img_buffer).decode('utf-8')


def post_image_from_path(url:str, path:str, shape:tuple[int,int]=None):
    img = cv2.imread(path)
    if not shape is None and isinstance(shape, tuple):
       img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    return post_image(url, img)   

def post_image(url:str, image:np.ndarray):
    headers= {
    'Content-Type': 'application/json',
    }
    response = requests.post(url, data=json.dumps({"body": image_to_base64(image)}), headers=headers)
    return response.json()
