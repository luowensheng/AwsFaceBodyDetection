
import opencv_stream 
import numpy as np
from model import Model, ModelOutput

stream = opencv_stream.VideoStreamer.from_webcam()
fps = opencv_stream.FpsDrawer()


model = Model()

@stream.on_next_frame()
def index(frame: np.ndarray):
   
   result = model.predict(frame) 

   if result.is_ok():
      output: ModelOutput = result.unwrap()
      output.draw(frame)

   fps.draw(frame)


stream.start()
