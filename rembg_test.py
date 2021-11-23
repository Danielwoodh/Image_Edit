from rembg.bg import remove
import numpy as np
import io
from PIL import Image, ImageFile

input_path = '042144-SEB.jpg'
output_path = 'out.png'

# Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
ImageFile.LOAD_TRUNCATED_IMAGES = True

f = np.fromfile(input_path)
print(type(f))
print(f.shape)
result = remove(f)
img = Image.open(io.BytesIO(result)).convert("RGBA")
img.save(output_path)