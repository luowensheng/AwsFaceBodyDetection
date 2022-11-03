
from opencv_stream import VideoStreamer, FpsDrawer
import numpy as np
from model import Model, ModelOutput

stream = VideoStreamer.from_webcam()
fps = FpsDrawer()

model = Model()

@stream.on_next_frame()
def index(frame: np.ndarray):
   
   result = model.predict(frame) 

   if result.is_ok():
      output: ModelOutput = result.unwrap()
      output.draw(frame)

   fps.draw(frame)


stream.start()
