from utils import post_image
import opencv_stream
import numpy as np

class Model(opencv_stream.Model):
   def __init__(self) -> None:
       self.api_url = "https://j4a9jaenyj.execute-api.ap-northeast-1.amazonaws.com/Prod/face/"
      #  self.api_url = "http://127.0.0.1:3000/face"
   
   @opencv_stream.Option.wrap
   def predict(self, image: np.ndarray) -> opencv_stream.Option:
      data = post_image(self.api_url, image) 
      return ModelOutput(data)

class ModelOutput(opencv_stream.ModelOutput):

   def __init__(self, data) -> None:
      self.data = data
      

   def to_dict(self) -> dict:
      return {"success": True}

   def draw(self, image: np.ndarray) -> None:
      print(self.data)