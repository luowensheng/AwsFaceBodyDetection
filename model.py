import math
from typing import Tuple, Union
from utils import post_image
import opencv_stream
import numpy as np
import json
import cv2

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


class Model(opencv_stream.Model):
   def __init__(self) -> None:
       self.api_url = "https://j4a9jaenyj.execute-api.ap-northeast-1.amazonaws.com/Prod/face/"
    #    self.api_url = "http://127.0.0.1:3000/face"
   
   @opencv_stream.Option.wrap
   def predict(self, image: np.ndarray) -> opencv_stream.Option:
      data = post_image(self.api_url, image) 
      return ModelOutput(data)

class ModelOutput(opencv_stream.ModelOutput):

   def __init__(self, data:dict) -> None:
    self.shape = data.get('shape')
    self.points: dict = data.get('data', {}).get('0', [])
    self.success: bool = data.get('success')
 
   def to_dict(self) -> dict:
      return {"success": True}

   def draw(self, image: np.ndarray) -> None:
       
       img_width, img_height, *_ = image.shape

       for point in self.points:

           center = _normalized_to_pixel_coordinates(point['x'], point['y'], img_height, img_width) 
           cv2.circle(image, center, 1, (255, 0, 0), 2) 

       return image      

           