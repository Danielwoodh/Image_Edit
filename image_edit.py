import os
from rembg.bg import remove
import numpy as np
import io
import glob
from PIL import Image, ImageFile
from tkinter import Tk, filedialog


class imageClean():
# Cleaning an image, resizing, removing background

	def main(self, filename: str, input_dir: str, output_dir: str, dim_x: int, dim_y: int, resize: str):
		input_img = np.fromfile(f'{input_dir}/{filename}')

		img_clean = self.clean(input_img)

		# Try with the new 'if' statement to see if it fixes the issues
		if resize == 'y':
			img_resize = self.resize(img_clean, dim_x, dim_y)
			img_rec = self.back_recolour(img_clean)
		elif resize == 'n':
			continue
		# img_rec = self.back_recolour(img_clean)

		self.save(img_clean, output_dir, filename)


	def clean(self, img):
		# Removing background from the image
		res = remove(img, alpha_matting=True)

		res_img = Image.open(io.BytesIO(res)).convert("RGBA")
		return res_img


	# Look for an alternative way to square the image (Pasting original image over a white background in PIL)
	def resize(self, img_res: str, dim_x: int, dim_y: int):
		# Finding current dimensions of the image
		height, width = img_res.size
		x = height if height > width else width
		y = height if height > width else width

		# Squaring the image
		if height != width:
			square = np.zeros((x, y, 4), np.uint8)
			square[int((x-width)/2):int(x-(x-width)/2), int((y-height)/2):int(y-(y-height)/2)] = img_res

			# Resizing the squared image
			square.resize((dim_x, dim_y))
			
			return square
		else:
			return img_res


	def back_recolour(self, img_recolour):
		fill = (255, 255, 255)

		# img_recolour_switch = Image.fromarray(img_recolour).convert('RGBA')
		# print(img_recolour_switch.mode)

		background = Image.new(img_recolour.mode[:-1], img_recolour.size, fill)
		background.paste(img_recolour, img_recolour)

		# FUNCTION TO REPLACE BLACK PIXELS WITH WHITE, WORKS BUT IS JANKY
		# newimdata = []
		# des_colour = (255, 255, 255)
		# for colour in background.getdata():
		# 	if colour == (0, 0, 0):
		# 		newimdata.append(des_colour)
		# 	else:
		# 		newimdata.append(colour)
		# newim = Image.new(background.mode, background.size)
		# newim.putdata(newimdata)

		return newim


	def save(self, final_img, output_dir: str, filename: str):
		folder = filename.split('-')

		file_split = filename.split('.')

		# final_img_conv = Image.fromarray(final_img).convert('RGBA')

		if not os.path.exists(f'{output_dir}/{folder[0]}'):
			os.makedirs(f'{output_dir}/{folder[0]}')

		final_img.convert('RGBA').save(f'{output_dir}/{folder[0]}/{file_split[0]}.png', 'PNG', optimize=True)


def open_folder():
    """
    Promps the user to select a folder, using a UI.
    """

    root = Tk()
    root.filename = filedialog.askdirectory()
    root.withdraw()
    return root.filename


if __name__ == '__main__':
	# Allows loading JPEG and JPG files
	ImageFile.LOAD_TRUNCATED_IMAGES = True

	# User-inputs for desired directories
	print('Select the input folder:')
	input_dir = open_folder()

	print('Select the output folder:')
	output_dir = open_folder()

	# User-inputs for desired dimensions
	dimension_x = int(input("Enter the desired image dimensions (X): "))
	dimension_y = int(input("Enter the deisred image dimensions (Y): "))

	resize = input("Do the images need background removal (y/n): ")

	# Instantiating imageClean class
	imgCleaner = imageClean()

	print(os.listdir(input_dir))

	for filename in os.listdir(input_dir):
		if not filename.startswith('.'):
			imgCleaner.main(filename, input_dir, output_dir, dimension_x, dimension_x, resize)
