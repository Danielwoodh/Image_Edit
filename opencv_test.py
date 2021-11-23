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

	def main(self, filename: str, input_dir: str, output_dir: str, dim_x: int, dim_y: int):
		'''
		Main file, calling the instantiated methods from within the imageClean class
		'''

		img = self.openFile(filename, input_dir)

		img_ero = self.canny(img)

		mask_stack, x, y, w, h = self.contours(img, img_ero)

		img_comb = self.combine(mask_stack, img)

		img_res = self.resize(img_comb, dim_x, dim_y, x, y, w, h)

		del img, img_ero, mask_stack, x, y, w, h, img_comb

		self.save(img_res, output_dir, filename)


	def openFile(self, filename: str, input_dir: str):
		img = cv2.imread(f'{input_dir}/{filename}', cv2.IMREAD_UNCHANGED)

		return img


	def canny(self, img):
		# Converting to grayscale
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# Blurring image, gives more defined boundaries for edge detection
		img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
		# Canny edge detection
		img_canny = cv2.Canny(img_blur, 100, 200, 3)

		# Dilating and eroding to fill blank areas
		img_dil = cv2.dilate(img_canny, None)
		img_ero = cv2.erode(img_dil, None)

		return img_ero


	def contours(self, img, img_ero):
		contour_info = []

		# Finding contours from the Canny image
		contours, _ = cv2.findContours(img_ero, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		for c in contours:
			contour_info.append((
				c,
				cv2.isContourConvex(c),
				cv2.contourArea(c)
			))

		contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
		max_contour = contour_info[0]

		x, y, w, h = cv2.boundingRect(max_contour[0])

		# Constructing a mask
		mask = 255*np.ones(img_ero.shape)

		cv2.fillConvexPoly(mask, max_contour[0], (255))

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


	def resize(self, img_res: str, dim_x: int, dim_y: int, x: int, y: int, w: int, h: int):

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
		# Finding current dimensions of the image
		dy = img.shape[0]
		dx = img.shape[1]

		# This assumes that the image is in the correct resolution already - Poor assumption to make
		desired_y = dy * 0.1
		desired_x = dx * 0.1

		# 
		cur_bottom_y = dy - (y + h)
		print(cur_bottom_y)
		curp_bottom_y = cur_bottom_y / dy
		cur_top_y = y
		curp_top_y = cur_top_y / dy
		curp_y = dy / (cur_bottom_y + cur_top_y)

		cur_left_x = x
		curp_left_x = cur_left_x / dx
		cur_right_x = (dx - x) - w
		curp_right_x = cur_right_x / dy
		curp_x = dx / (cur_left_x + cur_right_x)

		# =====================================================================================================

		# =================================== Increasing border size ==========================================
		if curp_y < 0.2 or curp_x < 0.2:
			bottom = 0
			top = 0
			left = 0
			right = 0

			desp_bottom_y = ((desired_y - cur_bottom_y) / cur_bottom_y)
			desp_top_y = ((desired_y - cur_top_y) / cur_top_y)

			desp_left_x = ((desired_x - cur_left_x) / cur_left_x)
			desp_right_x = ((desired_x - cur_left_x) / cur_left_x)

			if curp_bottom_y < 0.1:
				print('curp_bot_y calc')
				bottom = desired_y - cur_bottom_y
			if curp_top_y < 0.1:
				print('curp_top_y calc')
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

	# User-inputs for desired directories
	print('Select the input folder:')
	input_dir = open_folder()

	print('Select the output folder:')
	output_dir = open_folder()

	# User-inputs for desired dimensions
	dimension_x = int(input("Enter the desired image dimensions (X): "))
	dimension_y = int(input("Enter the deisred image dimensions (Y): "))
	print(os.listdir(input_dir))

	for filename in os.listdir(input_dir):
		if not filename.startswith('.'):
			cleaner.main(filename, input_dir, output_dir, dimension_x, dimension_x)
			gc.collect()
			print(filename)

	print('Finished editing!')
