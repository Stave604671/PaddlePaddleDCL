import PIL
import numpy as np
from PIL import Image, ImageStat
image = np.ones(shape=[100, 100, 3])
image_s = np.mean(image)