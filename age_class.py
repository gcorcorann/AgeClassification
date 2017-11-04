import numpy as np
import cv2
import matplotlib.pyplot as plt

""" Dyanmic Range """
def dynamic_range(img):
    """
    Sets the dynamic range of the image.

    @param  img:    input image

    @return d_img:  new image with dynamic range
    """
    fmin = img.min()
    fmax = img.max()
    d_img = ((img - fmin) / (fmax - fmin)) * 255
    return d_img

""" Apply Sobel """
def apply_sobel(img, ksize, thres=None):
    """
    Apply Sobel operator [ksizexksize] to image.

    @param  img:    input image
    @param  ksize:  Sobel kernel size
                    @pre odd integer >= 3
    @param  thres:  binary threshold, if None do not threshold
                    @pre integer >= 0 & <= 255
	
    @return:    image of Sobel magnitude
    """
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    Im = cv2.magnitude(Ix, Iy)
    if thres is not None:
        _, It = cv2.threshold(Im, thres, 1, cv2.THRESH_BINARY)
        return It
    else:
        return Im

""" Convolve 1D """
def convolve_1D(vector, kernel):
    """
    1D Convolution with kernel.

    @param  vector: 1D array to convolve
    @param  kernel: 1D convolution kernel

    @return:    resultant convolved vector
    """
    offset = len(kernel)//2
    vector_pad = np.zeros(len(vector) + offset*2)
    vector_pad[offset:-offset] = vector
    vector_filt = np.zeros_like(vector)
    for i in range(len(vector)):
    	total = sum(vector_pad[i:i+len(kernel)] * kernel)
    	vector_filt[i] = total
    return vector_filt

""" Find Zero Crossings """
def find_zero_crossings(vector):
    """
    Find zero crossings in 1D data.

    @param  vector: 1D data to find zero-crossings

    @return mouth_pos:  location of mouth in original image
    @return nose_pos:   location of nose in original image
    """
    val = []
    pos = []
    # finite difference
    for i in range(len(vector)-1):
        if ((vector[i] > 0 and vector[i+1] <0) or 
            (vector[i] <0 and vector[i+1] >0)):
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

    @param  img_bin:    binary input image
    @param  flag:   0 = eye image, 1 = nose+mouth image

    @return mouth_pos:  location of mouth position
    @return nose_pos:   location of nose position
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

    @param  img:    input image
    @param  width:  width of input image
    @param  height: height of input image
    @param  sobel_thres:    threshold of Sobel operator for binary image

    @return eye_pos:    position of eyes relative to original image
    @return left_eye:   bounding box of left eye
    @return right_eye:  bounding box of right eye
    """
    # eye ROI
    roi = img[height//3:3*height//5, width//4:3*width//4].copy()
    # apply Sobel operator
    I_edge = apply_sobel(roi, 3, sobel_thres)
    # find position of eyes
    eye_pos = horizontal_projection(I_edge, 0)
    # find area of eyes
    labels, stats = connected_components(np.uint8(I_edge))
    # threshold components
    best_left = np.inf
    best_right = np.inf
    # find closest blobs to eye_pos that has an area >= 130
    for i in range(1, len(stats)):
        row = stats[i]
        if row[4] >= 130:
            c_x = row[0] + row[2]//2
            c_y = row[1] + row[3]//2
            # find distance from eye position
            d = abs(eye_pos - (c_y+height//3))
            # left eye
            if c_x <= labels.shape[1]//2 and d < best_left:
                left_eye = row[0:4]
                best_left = d
            # right eye
            elif c_x >= labels.shape[1]//2 and d < best_right:
                right_eye = row[0:4]
                best_right = d
    # offset compared to original image
    eye_pos += height//3
    left_eye[0] += width//4
    left_eye[1] += height//3
    right_eye[0] += width//4
    right_eye[1] += height//3
    return eye_pos, left_eye, right_eye 

""" Connected Components """
def connected_components(img_thres):
    """
    Find connected components along with statistics.

    @param  img_thres:  input binary image

    @return labels: labeled connected components
    @return stats:  statistics on each connected component
    """
    # morphological closing
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img_close = cv2.morphologyEx(img_thres, cv2.MORPH_CLOSE, se)
    # find connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img_close, 8)
    return labels, stats

""" Locate Nose Mouth """
def locate_nose_mouth(img, left_eye, right_eye, sobel_thres):
    """
    Find position of the nose and mouth in original image.

    @param  img:    input image
    @param  left_eye:   position of left eye in original image
    @param  right_eye:  position of right eye in original image
    @param  sobel_thres:    threshold for creating binary edge image

    @return nose_pos:   location of nose in original image
    @return mouth_pos:  location of mouth in original image
    """
    # distance between eyes
    d_eyes = right_eye[0]+right_eye[2] - left_eye[0]
    # find nose+mouse region of interest
    tb = max(left_eye[1]+left_eye[3], right_eye[1]+right_eye[3])
    bb = tb + (4 * d_eyes)//4
    lb = left_eye[0] + left_eye[2]//4
    rb = right_eye[0] + 3*right_eye[2]//4
    # apply Sobel filters`
    roi = img[tb:bb, lb:rb].copy()
    I_edge = apply_sobel(roi, 3, sobel_thres)
    # find mouth and nose position
    mouth_pos, nose_pos = horizontal_projection(I_edge, 1)
    mouth_pos += tb
    nose_pos += tb
    # find area of mouth
    labels, stats = connected_components(np.uint8(I_edge))
    mouth_area = stats[np.argmax(stats[1:,4])+1]
    mouth_area = mouth_area[0:4]
    mouth_area[0] += lb
    mouth_area[1] += tb
    return nose_pos, mouth_pos, mouth_area

""" Location Phase """
def location_phase(img, sobel_thres):
    """
    Find location of eyes, nose, and mouth.

    @param  img:    input image

    @return eye_pos:    position of eyes center
    @return left_eye:   bounding box of left eye
    @return right_eye:  bounding box of right eye
    @return nose_pos:   position of nose
    @return mouth_pos:  position of mouth
    @return mouth_area: bounding box of mouth
    """
    # keep original image for displaying
    height, width = img.shape
    # find eyes
    eye_pos, left_eye, right_eye = locate_eyes(img, width, height, sobel_thres)
    # find nose and mouth
    nose_pos, mouth_pos, mouth_area = locate_nose_mouth(img, left_eye, 
                                        right_eye, sobel_thres)
    return eye_pos, left_eye, right_eye, nose_pos, mouth_pos, mouth_area

""" Wrinkle Density """
def wrinkle_density(img_roi, wrinkle_thres):
    """ Calculates the wrinkle desity in a given region of interest.

    @param  img_roi:    Sobel region of interest image
    @param  wrinkle_thres:  threshold to consider a wrinkle or not

    @return:    wrinkle density [0-1]
    """
    W = np.sum(img_roi >= wrinkle_thres)
    P = img_roi.shape[0] * img_roi.shape[1]
    return W / P

""" Wrinkle Depth """
def wrinkle_depth(img_roi, wrinkle_thres):
    """ Calculates the wrinkle dpeth in a given region of interest.

    @param  img_roi:    Sobel region of interest image
    @param  wrinkle_thres:  threshold to consider a wrinkle or not

    @return:    wrinkle depth
    """
    W_A = img_roi[img_roi >= wrinkle_thres]
    M = np.sum(W_A)
    return M / (255*W_A) 

""" Skin Variance """
def skin_variance(img_roi):
    """ Calculates the average skin variance in a given region of interest.

    @param  img_roi:    Sobel region of interest image

    @return:    average skin variance
    """
    M = np.sum(img_roi)
    P = img_roi.shape[0] * img_roi.shape[1]
    return M / (255*P)

""" Feature Extraction """
def feature_extraction(img, eye_pos, left_eye, right_eye, nose_pos,
    mouth_pos, mouth_area, wrinkle_thres):
    """ Extract wrinkle and geometric features from input image.
    """
    height, width = img.shape
    # find wrinkle image
    img_wrinkle = apply_sobel(img, 3)

    # define forehead region
    lb = mouth_area[0]
    rb = mouth_area[0] + mouth_area[2]
    bb = eye_pos - max(left_eye[3], right_eye[3]) 
    tb = bb - max(mouth_area[0]-left_eye[0], 
            right_eye[1]+right_eye[2]-mouth_area[0]-mouth_area[2])
    forehead = [lb, tb, rb-lb, bb-tb]
    # find forehead wrinkle density
    roi = img_wrinkle[tb:bb, lb:rb]
    D1_forehead = wrinkle_density(roi, wrinkle_thres)
    D2_forehead = wrinkle_depth(roi, wrinkle_thres)
    V_forehead = skin_variance(roi)

    # define left eye corner
    lb = left_eye[0] - max(left_eye[2], right_eye[2])//4
    rb = left_eye[0]
    tb = left_eye[1]
    bb = left_eye[1] + left_eye[3]
    left_c = [lb, tb, rb-lb, bb-tb]
    # find left eye corner wrinkle density
    roi = img_wrinkle[tb:bb, lb:rb]
    D1_left_corner = wrinkle_density(roi, wrinkle_thres)
    D2_left_corner = wrinkle_depth(roi, wrinkle_thres)
    V_left_corner = skin_variance(roi)

    # define right eye corner
    lb = right_eye[0] + right_eye[2]
    rb = lb + max(left_eye[2], right_eye[2])//4
    tb = right_eye[1]
    bb = right_eye[1] + right_eye[3]
    right_c = [lb, tb, rb-lb, bb-tb]
    # find right eye corner wrinkle density
    roi = img_wrinkle[tb:bb, lb:rb]
    D1_right_corner = wrinkle_density(roi, wrinkle_thres)
    D2_right_corner = wrinkle_depth(roi, wrinkle_thres)
    V_right_corner = skin_variance(roi)

    # define left cheek region
    lb = left_c[0] + left_c[2]
    rb = lb + (mouth_area[0] - left_eye[0])
    tb = eye_pos + min(left_eye[3], right_eye[3])//4
    bb = mouth_area[1]
    cheek_left = [lb, tb, rb-lb, bb-tb]
    # find left cheek wrinkle density
    roi = img_wrinkle[tb:bb, lb:rb]
    D1_left_cheeck = wrinkle_density(roi, wrinkle_thres)
    D2_left_cheek = wrinkle_depth(roi, wrinkle_thres)
    V_left_cheek = skin_variance(roi)

    # define right cheek region
    rb = right_c[0]
    lb = rb - (right_eye[0]+right_eye[2]-mouth_area[0]-mouth_area[2])
    tb = eye_pos + min(left_eye[3], right_eye[3])//4
    bb = mouth_area[1]
    cheek_right = [lb, tb, rb-lb, bb-tb]
    # find right cheek wrinkle density
    roi = img_wrinkle[tb:bb, lb:rb]
    D1_right_cheek = wrinkle_density(roi, wrinkle_thres)
    D2_right_cheek = wrinkle_depth(roi, wrinkle_thres)
    V_right_cheek = skin_variance(roi)

    # draw forehead region
    cv2.rectangle(img, (forehead[0],forehead[1]), (forehead[0]+forehead[2],
        forehead[1]+forehead[3]), (0,0,0), 2)
    cv2.rectangle(img, (left_c[0],left_c[1]), (left_c[0]+left_c[2],
        left_c[1]+left_c[3]), (0,0,0), 2)
    cv2.rectangle(img, (right_c[0],right_c[1]), (right_c[0]+right_c[2],
        right_c[1]+right_c[3]), (0,0,0), 2)
    cv2.rectangle(img, (cheek_left[0],cheek_left[1]), 
        (cheek_left[0]+cheek_left[2],
        cheek_left[1]+cheek_left[3]), (0,0,0), 2)
    cv2.rectangle(img, (cheek_right[0],cheek_right[1]), 
        (cheek_right[0]+cheek_right[2],cheek_right[1]+cheek_right[3]), 
        (0,0,0), 2)
    #plt.figure()
    #plt.subplot(122), plt.imshow(img, cmap='gray')
    #plt.title('Feature Extraction')

def main():
    import sys
    if len(sys.argv) >= 2:
    	# read image from command line
    	img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    else:
        # read default image
        img = cv2.imread('images/baby.png', cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, (380,480))
    img = dynamic_range(img)
    # create a copy for processing
    img_copy = img.copy()
    height, width = img.shape
    # find locations
    eye_pos, left_eye, right_eye, nose_pos, mouth_pos, \
        mouth_area = location_phase(img_copy, 130)
    feature_extraction(img_copy, eye_pos, left_eye, right_eye, nose_pos, 
                        mouth_pos, mouth_area, 40)

    # draw center line
    cl = width // 2
    cv2.line(img, (cl,0), (cl,height), (0,0,0), 2)
    # draw eye position
    cv2.line(img, (0,eye_pos), (width,eye_pos), (0,0,0), 2)
    # draw eye areas
    cv2.rectangle(img, (left_eye[0],left_eye[1]), (left_eye[0]+left_eye[2],left_eye[1]+left_eye[3]), (0,0,0), 2)
    cv2.rectangle(img, (right_eye[0],right_eye[1]), (right_eye[0]+right_eye[2],right_eye[1]+right_eye[3]), (0,0,0), 2)
    # draw nose position
    cv2.line(img, (0,nose_pos), (width,nose_pos), (0,0,0), 2)
    # draw mouth position
    cv2.line(img, (0,mouth_pos), (width,mouth_pos), (0,0,0), 2)
    # draw mouth area
    cv2.rectangle(img, (mouth_area[0],mouth_area[1]), (mouth_area[0]+mouth_area[2],mouth_area[1]+mouth_area[3]), (0,0,0), 2)
    # display eyes
    #plt.subplot(121), plt.imshow(img, cmap='gray')
    #plt.title('Display Image')
    #plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    main()

