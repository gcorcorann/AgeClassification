import numpy as np
import cv2
import matplotlib.pyplot as plt

""" Dyanmic Range """
def dynamic_range(img):
	"""
	Sets the dynamic range of the image.

	@param img: input image

	@return d_img: new image with dynamic range
	"""
	fmin = img.min()
	fmax = img.max()
	d_img = ((img - fmin) / (fmax - fmin)) * 255
	return d_img

""" Apply Sobel """
def apply_sobel(img, ksize, thres):
	"""
	Apply Sobel operator [ksizexksize] to image.

	@param img: input image
	@param ksize: Sobel kernel size
					@pre odd integer >= 3
	@param thres: binary threshold
					@pre integer >= 0 & <= 255
	
	@return It: threshold image of Sobel magnitude
	"""
	Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
	Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
	Im = cv2.magnitude(Ix, Iy)
	_, It = cv2.threshold(Im, thres, 1, cv2.THRESH_BINARY)
	return It

""" Convolve 1D """
def convolve_1D(vector, kernel):
	"""
	1D Convolution with kernel.

	@param	vector:	1D array to convolve
	@param	kernel:	1D convolution kernel

	@return	vector_filt:	convolved vector
	"""
	offset = len(kernel)//2
	vector_pad = np.zeros(len(vector) + offset*2)
	vector_pad[offset:-offset] = vector
	vector_filt = np.zeros_like(vector)
	for i in range(len(vector)):
		total = sum(vector_pad[i:i+len(kernel)] * kernel)
		vector_filt[i] = total
	return vector_filt

def find_zero_crossings(vector):
	val = []
	pos = []
	# finite difference
	for i in range(len(vector)-1):
		if (vector[i] > 0 and vector[i+1] <0) or (vector[i] <0 and
			vector[i+1] >0):
			pos.append(i)
			val.append(abs(vector[i+1] - vector[i]))
	pos = np.array(pos)
	val = np.array(val)
	# find position of highest value
	idx = np.argmax(val)
	mouth_pos = pos[idx]
	val[idx] = 0
	while (True):
		idx = np.argmax(val)
		nose_pos = pos[idx]
		if abs(nose_pos - mouth_pos) <= 20:
			val[idx] = 0
		else:
			break
	return mouth_pos, nose_pos	

""" Horizontal Projection """
def horizontal_projection(img_bin, flag):
	"""
	Apply horiztonal projection to binary image.

	@param	img_bin:	binary input image
	@param	flag:	0 = eye image, 1 = nose+mouth image

	@return	locs:	locations of features
	"""
	height, width = img_bin.shape
	h_proj = []
	for row in img_bin:
		h_proj.append(row.sum())
	if flag==0:
		return h_proj.index(max(h_proj))
	elif flag==1:
		h_proj = np.array(h_proj)
		# smooth projection
		avg_kernel = np.ones(5) / 5
		h_proj = convolve_1D(h_proj, np.ones(5)/5)
		# find derivative
		der = convolve_1D(h_proj, np.array([-1,-1,0,1,1]))
		mouth_pos, nose_pos = find_zero_crossings(der)
		return mouth_pos, nose_pos

""" Locate Eyes """
def locate_eyes(img, width, height, sobel_thres):
	"""
	Find locations of both eyes and eye center line.

	@param	img:	input image
	@param	width:	width of input image
	@param	height:	height of input image
	@param	sobel_thres:	threshold of Sobel operator for binary image

	@return	eye_pos:	position of eyes relative to original image
	@return	left_eye:	bounding box of left eye
	@return	right_eye:	bounding box of right eye
	"""
	# eye ROI
	roi = img[height//3:3*height//5, width//8:7*width//8].copy()
	# apply Sobel operator
	I_edge = apply_sobel(roi, 3, sobel_thres)
	# find position of eyes
	eye_pos = horizontal_projection(I_edge, 0)
	# find area of eyes
	labels, stats = connected_components(np.uint8(I_edge))
	# threshold components
	best_left = 0
	best_right = 0
	for i in range(1, len(stats)):
		row = stats[i]
		if (row[4] >= 100):
			c_x = row[0] + row[2]//2
			# left eye
			if (c_x <= labels.shape[1]//2 and row[4] >= best_left):
				left_eye = row[0:4]
				best_left = row[4]
			elif (c_x >= labels.shape[1]//2 and row[4] >= best_right):
				right_eye = row[0:4]
				best_right = row[4]
	# offset compared to original image
	eye_pos += height//3
	left_eye[0] += width//8
	left_eye[1] += height//3
	right_eye[0] += width//8
	right_eye[1] += height//3
	# find eye positions
	return eye_pos, left_eye, right_eye 

def connected_components(img_thres):
	# morphological closing
	se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	img_close = cv2.morphologyEx(img_thres, cv2.MORPH_CLOSE, se)
	# find connected components
	_, labels, stats, _ = cv2.connectedComponentsWithStats(img_close, 8)
	return labels, stats

def locate_nose_mouth(img, left_eye, right_eye, sobel_thres):
	# find nose+mouse region of interest
	d_eyes = right_eye[0]+right_eye[2] - left_eye[0]
	top_bound = max(left_eye[1]+left_eye[3], right_eye[1]+right_eye[3])
	bot_bound = top_bound + (4 * d_eyes)//4
	left_bound = left_eye[0] + left_eye[2]//4
	right_bound = right_eye[0] + 3*right_eye[2]//4
	# apply Sobel filters`
	roi = img[top_bound:bot_bound, left_bound:right_bound].copy()
	I_edge = apply_sobel(roi, 3, sobel_thres)
	# find mouth and nose position
	mouth_pos, nose_pos = horizontal_projection(I_edge, 1)
	mouth_pos += top_bound
	nose_pos += top_bound
	# find area of mouth
	labels, stats = connected_components(np.uint8(I_edge))
	mouth_area = stats[np.argmax(stats[1:,4])+1]
	mouth_area = mouth_area[0:4]
	mouth_area[0] += left_bound
	mouth_area[1] += top_bound
	return nose_pos, mouth_pos, mouth_area

def location_phase(img):
	# keep original image for displaying
	img_disp = img.copy()
	height, width = img.shape
	# find eyes
	sobel_thres = 100
	eye_pos, left_eye, right_eye = locate_eyes(img_disp, width, height,
		sobel_thres)

	# find nose and mouth
	nose_pos, mouth_pos, mouth_area = locate_nose_mouth(img, left_eye,
		right_eye, 90)
	
	# draw nose position
	cv2.line(img_disp, (0,nose_pos), (width,nose_pos), (0,0,0), 2)
	# draw mouth position
	cv2.line(img_disp, (0,mouth_pos), (width,mouth_pos), (0,0,0), 2)
	# draw mouth area
	cv2.rectangle(img_disp, (mouth_area[0],mouth_area[1]),
		(mouth_area[0]+mouth_area[2],mouth_area[1]+mouth_area[3]), (0,0,0), 2)

	# draw center line
	cl = width // 2
	cv2.line(img_disp, (cl,0), (cl,height), (0,0,0), 2)
	# draw eye positions
	cv2.line(img_disp, (0,eye_pos), (width,eye_pos), (0,0,0), 2)
	cv2.rectangle(img_disp, (left_eye[0],left_eye[1]),
		(left_eye[0]+left_eye[2],left_eye[1]+left_eye[3]), (0,0,0), 2)
	cv2.rectangle(img_disp, (right_eye[0],right_eye[1]),
		(right_eye[0]+right_eye[2],right_eye[1]+right_eye[3]), (0,0,0), 2)
	# display eyes
	plt.figure()
	plt.subplot(121), plt.imshow(img, cmap='gray')
	plt.title('Input Image')
	plt.subplot(122), plt.imshow(img_disp, cmap='gray')
	plt.title('Display Image')
	plt.show()



# read image
img = cv2.imread('img1.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (380,480))

img = dynamic_range(img)
location_phase(img)
