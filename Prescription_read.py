import cv2
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# Helper Func For Later Stages
def nothing(x):
    pass


def Threshold_Demo(val):
    global res
    # 0: Binary
    # 1: Binary Inverted
    # 2: Threshold Truncated
    # 3: Threshold to Zero
    # 4: Threshold to Zero Inverted
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    _, dst = cv2.threshold(gray, threshold_value, max_binary_value, threshold_type)
    cv2.imshow(window_name, dst)
    res = dst


# Load the image and Convert to Gray Scale for Faster Computations
try:
    img=cv2.imread('PATH to the image')
    # Convert to Gray Scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 3)
    # sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # gray = cv2.filter2D(gray, -1, sharpen_kernel)
    max_value = 255
    max_type = 4
    max_binary_value = 255
    trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
    trackbar_value = 'Value'
    window_name = 'Threshold Demo'

    # Create TrackBars that will help us threshold for better Results.
    cv2.namedWindow(window_name)
    # Create Track bar to choose type of Threshold
    cv2.createTrackbar(trackbar_type, window_name, 0, max_type, Threshold_Demo)
    # Create Track bar to choose Threshold value
    cv2.createTrackbar(trackbar_value, window_name, 143, max_value, Threshold_Demo)

    # HERE WE WILL CLEAN THE IMAGE USING THRESHOLDING AND FILTERING
    res = img.copy()
    Threshold_Demo(0)
    cv2.waitKey(0)
    # Standard Thresholding to Make text Extraction Easier
    # _, th = cv2.threshold(res, 140, 255, cv2.THRESH_BINARY)
    th=res
    s = pytesseract.image_to_string(th)
    print('The Image reads :\n', s)
    num = [str(x) for x in range(10)]
    presciption_number = []
    for x in s:
        if(x in num):
            presciption_number.append(x)
    code=''.join(presciption_number)
    if code !="":
        print(f"The Prescription Number is {code}\n")
    h = img.shape
    numpy_vertical_concat = np.concatenate((th, res), axis=0)
    cv2.imshow('Results After Thresholding', numpy_vertical_concat)
    cv2.waitKey(0)
except:
    print("There Was Some Error!\nTry Again")

