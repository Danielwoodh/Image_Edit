		# =======================================================================================================
			# print('Border Creation')
			# if cur_y > cur_x:
			# 	if dy > dim_y:
			# 		desp_y = abs(curp_y - 0.1) / ((curp_y + 0.1) / 2)
			# 		# abs((0.1 - curp_y) / curp_y)

			# 		top = int(desp_y * img.shape[0])
			# 		print(f'top: {top}')
			# 		bottom = top
			# 		dy = dy + (top * 2)

			# 		img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, (255, 255, 255))
			# 		return img

			# 	elif dy <= dim_y:
			# 		desp_y = abs(curp_y - 0.1) / ((curp_y + 0.1) / 2)

			# 		top = int(desp_y * img.shape[0])
			# 		bottom = top
			# 		dy = dy + (top * 2)
			# 		ratio = dim_y / float(dy)

			# 		img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, (255, 255, 255))
			# 		img = cv2.resize(img, (int(dim_x*ratio), dim_y), interpolation=cv2.INTER_CUBIC)
			# 		return img

			# elif cur_y < cur_x:
			# 	if dx > dim_x:
			# 		desp_x = abs(curp_x - 0.1) / ((curp_x + 0.1) / 2)

			# 		left = int(desp_x * img.shape[1])
			# 		right = left
			# 		dx = dx + (left * 2)

			# 		img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255))
			# 		return img

			# 	elif dx <= dim_x:
			# 		desp_x = abs(curp_x - 0.1) / ((curp_x + 0.1) / 2)
			# 		# abs((0.1 - curp_x) / curp_x)

			# 		left = int(desp_x * img.shape[1])
			# 		right = left
			# 		dx = dx + (left * 2)
			# 		ratio = dim_x / float(dx)

			# 		img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255))
			# 		img = cv2.resize(img, (dim_x, int(dim_y*ratio)), interpolation=cv2.INTER_CUBIC)
			# 		return img

		# =======================================================================================================

		# ======================================= Decreasing border size ========================================
		# elif curp_y > 0.1 and curp_x > 0.1:
		# 	if cur_y < cur_x:
		# 		desp_y = abs(curp_y - 0.1) / ((curp_y + 0.1) / 2)
		# 		# abs((curp_y - 0.1) / 0.1)
		# 		print(f'desp_y: {desp_y}')
		# 		pix_reduce = int(cur_y - ((desp_y * cur_y) / 2))

		# 		img = img[0:(dy - pix_reduce), 0:(dx - pix_reduce)]
		# 		print(f'dx-pixred {dx - pix_reduce}')
		# 		dy = dy - pix_reduce
		# 		dx = dx - pix_reduce
		# 		img = img[(0 + pix_reduce):dy, (0 + pix_reduce):dx]
		# 		dy = dy - pix_reduce
		# 		dx = dx - pix_reduce
		# 		if dy > dim_y:
		# 			if dx == dy:
		# 				img = cv2.resize(img, (dim_x, dim_y), interpolation=cv2.INTER_AREA)
		# 				return img

		# 			else:
		# 				ratio = dim_y / float(dy)
		# 				img = cv2.resize(img, (int(dim_x*ratio), dim_y), interpolation=cv2.INTER_AREA)
		# 				return img
						
				# elif dy <= dim_y:
				# 	if dx == dy:
				# 		img = cv2.resize(img, (dim_x, dim_y), interpolation=cv2.INTER_CUBIC)
				# 		return img

				# 	else:
				# 		ratio = dim_y / float(dy)
				# 		img = cv2.resize(img, (int(dim_x*ratio), dim_y), interpolation=cv2.INTER_CUBIC)
				# 		return img

		# 	elif cur_y >= cur_x:
		# 		desp_x = abs(curp_x - 0.1) / ((curp_x + 0.1) / 2)
		# 		# abs((curp_x - 0.1) / 0.1)
		# 		print(f'desp_x: {desp_x}')
		# 		pix_reduce = int(cur_x - ((desp_x * cur_x) / 2))
		# 		print(f'pix_reduce: {pix_reduce}')

		# 		img = img[0:(dy - pix_reduce), 0:(dx - pix_reduce)]
		# 		dy = dy - pix_reduce
		# 		dx = dx - pix_reduce
		# 		img = img[(0 + pix_reduce):dy, (0 + pix_reduce):dx]
		# 		dy = dy - pix_reduce
		# 		dx = dx - pix_reduce

		# 		if dx > dim_x:
		# 			if dx == dy:
		# 				img = cv2.resize(img, (dim_x, dim_y), interpolation=cv2.INTER_AREA)
		# 				return img

		# 			else:
		# 				ratio = dim_x / float(dx)
		# 				img = cv2.resize(img, (dim_x, int(ratio*dim_y)), interpolation=cv2.INTER_AREA)
		# 				return img

		# 		elif dx <= dim_x:
		# 			if dx == dy:
		# 				img = cv2.resize(img, (dim_x, dim_y), interpolation=cv2.INTER_CUBIC)
		# 				return img

		# 			else:
		# 				ratio = dim_x / float(dx)
		# 				img = cv2.resize(img, (dim_x, int(ratio*dim_y)), interpolation=cv2.INTER_AREA)
		# 				return img
		# 	elif curp_y < 0.1 and curp_x > 0.1:
		# 		if dy > dim_y:
		# 			desp_y = abs(curp_y - 0.1) / ((curp_y + 0.1) / 2)
		# 			# abs((0.1 - curp_y) / curp_y)

		# 			top = int(desp_y * img.shape[0])
		# 			print(f'top: {top}')
		# 			bottom = top
		# 			dy = dy + (top * 2)

		# 			img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, (255, 255, 255))
		# 			return img

		# 		elif dy <= dim_y:
		# 			desp_y = abs(curp_y - 0.1) / ((curp_y + 0.1) / 2)

		# 			top = int(desp_y * img.shape[0])
		# 			bottom = top
		# 			dy = dy + (top * 2)
		# 			ratio = dim_y / float(dy)

		# 			img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, (255, 255, 255))
		# 			img = cv2.resize(img, (int(dim_x*ratio), dim_y), interpolation=cv2.INTER_CUBIC)
		# 			return img

		# 	elif curp_y > 0.1 and curp_x < 0.1:
		# 		if dx > dim_x:
		# 			desp_x = abs(curp_x - 0.1) / ((curp_x + 0.1) / 2)

		# 			left = int(desp_x * img.shape[1])
		# 			right = left
		# 			dx = dx + (left * 2)

		# 			img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255))
		# 			return img

		# 		elif dx <= dim_x:
		# 			desp_x = abs(curp_x - 0.1) / ((curp_x + 0.1) / 2)
		# 			# abs((0.1 - curp_x) / curp_x)

		# 			left = int(desp_x * img.shape[1])
		# 			right = left
		# 			dx = dx + (left * 2)
		# 			ratio = dim_x / float(dx)

		# 			img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255))
		# 			img = cv2.resize(img, (dim_x, int(dim_y*ratio)), interpolation=cv2.INTER_CUBIC)
		# 			return img
		# 	else:
		# 		return img
			# =================================================================================================

