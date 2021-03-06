'''
Daniel Woodhall - OpenCV Image resizing/border-removal
This script takes a series of input images and a desired border size (X, Y), to convert the inputs to squared
images with the desired border size. Images should be output in a uniform manner.

Notes/Potential issues:
- If there are more than 3 objects of importance within the input image, the script will not work as intended.
However, this could be changed if necessary

- If the input image is a smaller resolution than the desired output image it will be increased in size, which may
result in loss of image fidelity.

- As the output image is required to be JPEG, converting from other image formats (eg. PNG) will cause some loss of
fidelity due to the conversion from a lossless file format (PNG) to a lossy file format (JPEG). Ideally, the website
should be able to handle .PNG files as this would eliminate this; although it is not necessary.
'''

import cv2
import numpy as np
import io
import os
from PIL import Image
from tkinter import Tk, filedialog
import gc


class imageClean():
	'''
	This class encompasses the methods to remove the background from images
	'''

	def main(self, errors: list, filename: str, input_dir: str, output_dir: str, dim_x: int, dim_y: int, insensitivity: float):
		'''
		Main file, calling the instantiated methods from within the imageClean class
		'''

		img = self.openFile(filename, input_dir)

		img_ero, img_blur = self.canny(img)

		mask_stack, x, y, w, h = self.contours(img_blur, img_ero, insensitivity)

		img_comb = self.combine(mask_stack, img)

		img_res = self.resize(filename, img_comb, dim_x, dim_y, x, y, w, h)

		del img, img_ero, mask_stack, x, y, w, h, img_comb

		self.save(img_res, output_dir, filename)

		return errors


	def openFile(self, filename: str, input_dir: str):
		# Opens the files using cv2
		img = cv2.imread(f'{input_dir}/{filename}', cv2.IMREAD_UNCHANGED)

		return img


	def canny(self, img):
		'''
		Function to convert image and detect edges using Canny.
		'''
		# Converting to grayscale
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# Blurring image, gives more defined boundaries for edge detection
		img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

		# Canny edge detection
		img_canny = cv2.Canny(img_blur, 85, 255, 3)

		# Dilating and eroding to fill blank areas in edge mesh
		img_dil = cv2.dilate(img_canny, None)
		img_ero = cv2.erode(img_dil, None)

		return img_ero, img_blur


	def contours(self, img_blur, img_ero, insensitivity: float):
		# Constructing contours from the binary edge detected image
		contour_info = []

		# Finding contours from the Canny image, TEST WITH RETR_EXTERNAL AND RETR_LIST
		contours, _ = cv2.findContours(img_ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		for c in contours:
			contour_info.append((
				c,
				cv2.isContourConvex(c),
				cv2.contourArea(c)
			))

		contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

		max_contour = contour_info[0]

		x, y, w, h = cv2.boundingRect(max_contour[0])

		cv2.drawContours(img_ero, max_contour[0], 0, (0, 255, 0), thickness=10)

		# Checking if there are two substantial contours (such as images with pairs of items)
		if len(contour_info) > 1:
			if contour_info[1][2] >= (insensitivity * contour_info[0][2]):
				# Calculating a new bounding rectangle to encompass both contours
				x2, y2, w2, h2 = cv2.boundingRect(contour_info[1][0])

				min_y, max_y = min(y, y2), max((y+h), (y2+h2))
				min_x, max_x = min(x, x2), max((x+w), (x2+w2))
				w = max_x - min_x
				h = max_y - min_y
				x = min_x
				y = min_y

		# Constructing a mask
		mask = 255*np.ones(img_ero.shape)

		mask = cv2.dilate(mask, None, iterations=10)
		mask = cv2.erode(mask, None, iterations=10)

		mask = cv2.GaussianBlur(mask, (3, 3), 0)

		mask_stack = np.dstack([mask]*3)

		return mask_stack, x, y, w, h


	def combine(self, mask_stack, image):
		# Overlaying mask with original image to remove background
		if image.shape[2] == 4:
			trans_mask = image[:,:,3] == 0
			image[trans_mask] = [255, 255, 255, 255]
			image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

		mask_stack = mask_stack.astype('float32') / 255.0
		image = image.astype('float32') / 255.0

		masked = (mask_stack * image) 
		# + ((1 - mask_stack) * (255, 255, 255))

		masked = (masked * 255).astype('uint8')

		return masked


	def resize(self, filename: str, img_res: str, dim_x: int, dim_y: int, x: int, y: int, w: int, h: int):

		if y == 0 or x == 0:
			errors.append(filename)
			y += 1
			x += 1

		img_border = border_remove(img_res, dim_x, dim_y, x, y, w, h)
		img_square = square_image(img_border, dim_x, dim_y)
		return img_square


	def save(self, final_img, output_dir: str, filename: str):

		folder = filename.split('-')

		file_split = filename.split('.')

		if not os.path.exists(f'{output_dir}/{folder[0]}'):
			os.makedirs(f'{output_dir}/{folder[0]}')

		cv2.imwrite(f'{output_dir}/{folder[0]}/{file_split[0]}.jpg', final_img)

		del final_img


def open_folder():
    """
    Promps the user to select a folder, using a UI.
    """

    root = Tk()
    root.filename = filedialog.askdirectory()
    root.withdraw()
    return root.filename


# FIX IMAGES NOT BEING RESIZED / SQUARED PROPERLY
def border_remove(img, dim_x: int, dim_y: int, x: int, y: int, w: int, h: int):
		# Finding current dimensions of the image - CORRECT
		dy = img.shape[0]
		dx = img.shape[1]

		# Calculating desired border size
		desired_y = int(dy * 0.1)
		desired_x = int(dx * 0.1)

		# Calculating current border size (Y)
		cur_bottom_y = dy - (y + h)
		curp_bottom_y = cur_bottom_y / dy
		cur_top_y = y
		curp_top_y = cur_top_y / dy
		curp_y = (cur_bottom_y + cur_top_y) / dy

		# Calculating current border size (X)
		cur_left_x = x
		curp_left_x = cur_left_x / dx
		cur_right_x = (dx - x) - w
		curp_right_x = cur_right_x / dy
		curp_x = (cur_left_x + cur_right_x) / dx

		# =====================================================================================================
		# =================================== Increasing border size ==========================================
		# Checking if the current border size is smaller than the desired
		if curp_y < 0.2 or curp_x < 0.2:
			bottom = 0
			top = 0
			left = 0
			right = 0

			desp_bottom_y = ((desired_y - cur_bottom_y) / cur_bottom_y)
			desp_top_y = ((desired_y - cur_top_y) / cur_top_y)

			desp_left_x = ((desired_x - cur_left_x) / cur_left_x)
			desp_right_x = ((desired_x - cur_left_x) / cur_left_x)

			# Calculating amount of border to add in each direction
			if curp_bottom_y < 0.1:
				bottom = desired_y - cur_bottom_y
			if curp_top_y < 0.1:
				top = desired_y - cur_top_y

			if curp_left_x < 0.1:
				left = desired_x - cur_left_x
			if curp_right_x < 0.1:
				right = desired_x - cur_right_x

			img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255))

			if dy <= dim_y and dx > dim_x:
				ratio = dim_y / float(dy)
				img = cv2.resize(img, (int(dx* ratio), dim_y), interpolation=cv2.INTER_CUBIC)
				return img

			elif dy > dim_y and dx <= dim_x:
				ratio = dim_x / float(dx)
				img = cv2.resize(img, (dim_x, int(dy * ratio)), interpolation=cv2.INTER_CUBIC)
				return img

			elif dy <= dim_y and dx <= dim_x:
				dif_x = dim_x - dx
				dif_y = dim_y - dy

				if dif_y > dif_x:
					ratio = dim_y / float(dy)
					img = cv2.resize(img, (int(dx * ratio), dim_y), interpolation=cv2.INTER_CUBIC)
					return img

				else:
					ratio = dim_x / float(dx)
					img = cv2.resize(img, (dim_x, int(dy * ratio)), interpolation=cv2.INTER_CUBIC)
					return img

			elif dy > dim_y and dx > dim_x:
				ratio = dim_y / float(dy)
				img = cv2.resize(img, (int(dx * ratio), dim_y), interpolation=cv2.INTER_AREA)
				return img
		# ======================================= Decreasing border size ========================================
		# If border sizes are above a certain value, decrease the current border size
		elif curp_y > 0.2 and curp_x > 0.2:
			pix_reduce_top = 0
			pix_reduce_bottom = 0

			pix_reduce_right = 0
			pix_reduce_left = 0

			desp_top_y = ((cur_top_y - desired_y) / cur_top_y)
			desp_bottom_y = ((cur_bottom_y - desired_y) / cur_bottom_y)

			desp_left_x = ((cur_left_x - desired_x) / cur_left_x)
			desp_right_x = ((cur_right_x - desired_x) / cur_right_x)

			if curp_top_y > 0.1:
				pix_reduce_top = int(desp_top_y * cur_top_y)
			if curp_bottom_y > 0.1:
				pix_reduce_bottom = int(desp_bottom_y * cur_bottom_y)

			if curp_left_x > 0.1:
				pix_reduce_left = int(desp_left_x * cur_left_x)
			if curp_right_x > 0.1:
				pix_reduce_right = int(desp_right_x * cur_right_x)

			img = img[0:(dy - pix_reduce_bottom), 0:(dx - pix_reduce_right)]
			dx = dx - pix_reduce_right
			dy = dy - pix_reduce_bottom

			img = img[pix_reduce_top:dy, pix_reduce_left:dx]
			dx = dx - pix_reduce_left
			dy = dy - pix_reduce_top

			if dy > dim_y:
				if dx == dy:
					img = cv2.resize(img, (dim_x, dim_y), interpolation=cv2.INTER_AREA)
					return img

				else:
					ratio = dim_y / float(dy)
					img = cv2.resize(img, (int(dx*ratio), dim_y), interpolation=cv2.INTER_AREA)
					return img

			elif dy <= dim_y:
				if dx == dy:
					img = cv2.resize(img, (dim_x, dim_y), interpolation=cv2.INTER_CUBIC)
					return img
				else:
					ratio = dim_y / float(dy)
					img = cv2.resize(img, (int(dx * ratio), dim_y), interpolation=cv2.INTER_CUBIC)
					return img

			return img


def square_image(img, dim_x: int, dim_y: int):
	# Squaring the image whilst maintaining aspect ratio
	dy = img.shape[0]
	dx = img.shape[1]
	# Squaring the image
	if dx != dy and dx < dim_x and dy < dim_y:
		square_img = Image.new('RGB', (dim_x, dim_y), (255, 255, 255))
		img_pil = Image.fromarray(img)
		square_img.paste(img_pil, (int((dim_x - dx) / 2), int((dim_y - dy) / 2)))
		
		square_img_np = np.asarray(square_img)
		return square_img_np

	elif dx != dy and dy > dx:
		r = dim_y / float(dy)
		dim = (int(dx * r), dim_y)
		img = cv2.resize(img, dim)

		img_pil = Image.fromarray(img)
		square_img = Image.new('RGB', (dim_x, dim_y), (255, 255, 255))
		square_img.paste(img_pil, (int((dim_x - int(dx * r)) / 2), int((dim_y - dim_y) / 2)))
		square_img_np = np.asarray(square_img)
		return square_img_np

	elif dx != dy and dx > dy:
		r = dim_x / float(dx)
		dim = (dim_x, int(dy * r))
		img = cv2.resize(img, dim)

		img_pil = Image.fromarray(img)
		square_img = Image.new('RGB', (dim_x, dim_y), (255, 255, 255))
		square_img.paste(img_pil, (int((dim_x - dim_x) / 2), int((dim_y - int(dy * r)) / 2)))
		square_img_np = np.asarray(square_img)
		return square_img_np

	else:
		img = cv2.resize(img, (dim_x, dim_y))
		return img


if __name__ == '__main__':
	cleaner = imageClean()

	errors = []

	# User-inputs for desired directories
	print('Select the input folder:')
	input_dir = open_folder()

	print('Select the output folder:')
	output_dir = open_folder()

	# User-inputs for desired dimensions
	dimension_x = int(input("Enter the desired image dimensions (X): "))
	dimension_y = int(input("Enter the deisred image dimensions (Y): "))
	print("\nThe insensitivity must be between 0 and 1, with 0.3 being recommended.\nHigher resolution images (or images with multiple items) may require a lower value.")
	insensitivity = float(input("Enter the desired insensitivity: "))
	print(os.listdir(input_dir))

	for filename in os.listdir(input_dir):
		if not filename.startswith('.'):
			print(filename)
			cleaner.main(errors, filename, input_dir, output_dir, dimension_x, dimension_x, insensitivity)
			gc.collect()

	print('\nFinished editing!')
	print(f'\nError editing files: {errors}')