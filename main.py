""" Main function """
def main():
    """
    Reads all images in face dataset and applies location stage to find eyes,
    nose, and mouth.

    Call python script with two arguments: (1) face dataset, (2) sobel
    threshold in that order.

    """
    import sys
    if len(sys.argv) < 3:
        print("Please list database and sobel threshold.")
        return
    import numpy as np
    import cv2
    import glob
    import age_class
    # store all image paths in list
    img_list = glob.glob(sys.argv[1] + '*')
    # threshold for edge detection
    sobel_thres = int(sys.argv[2])
    feature_list = []
    for i in img_list:
        print(i)
        if i != "face_dataset/.jpg":
            img_color = cv2.imread(i)
            img_color = cv2.resize(img_color, (150, 200))
            height, width = 200, 150
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            img_gray = age_class.dynamic_range(img_gray)

            # location phase
            locs = age_class.location_phase(img_gray.copy(), sobel_thres)
            eye_pos, left_eye, right_eye, nose_pos, mouth_pos, mouth_area = locs
            # feature extraction phase
            feats = age_class.feature_extraction(img_gray.copy(), eye_pos, 
                    left_eye, right_eye, nose_pos, mouth_pos, mouth_area, 40)
            # place all feature data into feature matrix
            l = []
            for n in range(len(feats[0])):
                for key in feats[0][n].keys():
                    l.append(feats[0][n][key])
            l.append(feats[1])
            l.append(feats[2])
            feature_list.append(l)

    feature_list = np.array(feature_list)
    print(feature_list.shape)
if __name__ == "__main__":
    print(main.__doc__)
    main()
