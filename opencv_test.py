import cv2
import numpy as np
import io
import os
from PIL import Image
from tkinter import Tk, filedialog


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
		# Finding current dimensions of the image
		dy = img_res.shape[0]
		dx = img_res.shape[1]
		print(y)
		print(dy)
		cur_y = dy - (y + h)
		curp_y = cur_y / dy
		print(curp_y)

		# IMPROVE BORDER CALCULATIONS - THIS IS CHANGING ASPECT RATIO - ADD ALL OF THIS TO A SEPARATE FUNCTION
		if curp_y < 0.075 and dy > dim_y:
			diff_y = 0.075 - curp_y
			top = int(diff_y * img_res.shape[0])
			bottom = top
			dy = dy + (top * 2)
			img_res = cv2.copyMakeBorder(img_res, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, (255, 255, 255))
		elif curp_y > 0.075 and dy <= dim_y:
			desp_y = abs((0.075 - curp_y) / curp_y)
			print(desp_y)

			pix_reduce = int(cur_y - ((desp_y * cur_y) / 2))
			print(f'pix reduce: {pix_reduce}')

			img_res = img_res[0:(dy - pix_reduce), :]
			dy = dy - pix_reduce
			img_res = img_res[(0 + pix_reduce):dy, :]
			dy = dy - pix_reduce

			ratio = dim_y / float(dy)
			img_res = cv2.resize(img_res, (int(dim_x*ratio), dim_y))
			print(f'dy2: {dy}')
			cv2.imshow('image', img_res)
			cv2.waitKey(0)
		# elif curp_y < 0.075 and dy < dim_y:

		# elif curp_y > 0.075 and dy > dim_y:


		# Squaring the image
		if dx != dy and dx < dim_x and dy < dim_y:
			square_img = Image.new('RGB', (dim_x, dim_y), (255, 255, 255))
			img_pil = Image.fromarray(img_res)
			square_img.paste(img_pil, (int((dim_x - dx) / 2), int((dim_y - dy) / 2)))
			
			square_img_np = np.asarray(square_img)
			return square_img_np

		elif dx != dy and dy > dx:
			r = dim_y / float(dy)
			dim = (int(dx * r), dim_y)
			img_res = cv2.resize(img_res, dim)

			img_pil = Image.fromarray(img_res)
			square_img = Image.new('RGB', (dim_x, dim_y), (255, 255, 255))
			square_img.paste(img_pil, (int((dim_x - int(dx * r)) / 2), int((dim_y - dim_y) / 2)))
			square_img_np = np.asarray(square_img)

			return square_img_np

		elif dx != dy and dx > dy:
			r = dim_x / float(dx)
			dim = (dim_x, int(dy * r))
			img_res = cv2.resize(img_res, dim)

			img_pil = Image.fromarray(img_res)
			square_img = Image.new('RGB', (dim_x, dim_y), (255, 255, 255))
			square_img.paste(img_pil, (int((dim_x - dim_x) / 2), int((dim_y - int(dy * r)) / 2)))
			square_img_np = np.asarray(square_img)

			# square_img_np = cv2.resize(square_img_np, (dim_x, dim_y))

			return square_img_np
		else:
			img_res = cv2.resize(img_res, (dim_x, dim_y))
			return img_res


	def save(self, final_img, output_dir: str, filename: str):
		folder = filename.split('-')

		file_split = filename.split('.')

		if not os.path.exists(f'{output_dir}/{folder[0]}'):
			os.makedirs(f'{output_dir}/{folder[0]}')

		cv2.imwrite(f'{output_dir}/{folder[0]}/{file_split[0]}.jpg', final_img)


def open_folder():
    """
    Promps the user to select a folder, using a UI.
    """

    root = Tk()
    root.filename = filedialog.askdirectory()
    root.withdraw()
    return root.filename


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

	for filename in os.listdir(input_dir):
		if not filename.startswith('.'):
			cleaner.main(filename, input_dir, output_dir, dimension_x, dimension_x)