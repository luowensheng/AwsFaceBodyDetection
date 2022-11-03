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

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
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
       if self.shape is None: return

       img_width, img_height = self.shape
       for point in self.points:
           x, y, z = [point[axis] for axis in ['x', 'y', 'z']]
        #    center = (x, y)
           center = _normalized_to_pixel_coordinates(x, y, img_width, img_height) 
           cv2.circle(image, center, 1, (255, 0,0), 2)   

           