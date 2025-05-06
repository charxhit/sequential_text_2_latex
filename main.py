

import cv2
import numpy as np
from functools import cmp_to_key


class ImagePreProcessing:

    def __init__(self):
        pass

    def extract_sym(self, original_img, show_steps=False, vertical_sym=False):
        debug_steps = []
        img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img_filtered = cv2.medianBlur(img_gray, 5)
        debug_steps.append(img_filtered)

        img_canny = cv2.Canny(img_filtered, 50, 180)
        debug_steps.append(img_canny)

        kernel = np.ones((5, 5), np.uint8)
        img_dilated = cv2.dilate(img_canny, kernel, iterations=5)
        debug_steps.append(img_dilated)

        contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

        global rowsG
        rowsG, _, _ = original_img.shape
        key_leftRightTopBottom = cmp_to_key(leftRightTopBottom)
        if (vertical_sym):
            bounding_boxes = sorted(bounding_boxes, key=key_leftRightTopBottom)
        else:
            bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])

        symbols = []
        for box in bounding_boxes:
            x, y, w, h = box
            math_symbol = original_img[y:y + h, x:x + w]
            math_symbol = cv2.cvtColor(math_symbol,
                                       cv2.COLOR_BGR2GRAY)  # converting to Gray as tensorflow deals with grayscale or RGB, not BGR
            math_symbol = cv2.resize(math_symbol, (45, 45),
                                     interpolation=cv2.INTER_AREA)  # to have the same size as trained images in the dataset
            debug_steps.append(math_symbol)
            math_symbol_f = math_symbol.astype('float32')  # optional: tensorflows deals with float32, not uint8
            symbols.append(math_symbol_f)

        if show_steps:
            dispImages(debug_steps)

        return symbols

def leftRightTopBottom(tup1, tup2):
    x1, y1, _, _ = tup1
    x2, y2, _, _ = tup2
    rows = rowsG
    yRegion1, yRegion2 = -1, -1

    for i in range(4):
        if y1 < rows / 4 + rows * (i / 4.0):
            yRegion1 = i
            break
    else:
        if yRegion1 == -1:
            yRegion1 = 4

    for i in range(4):
        if y2 < rows / 4 + rows * (i / 4.0):
            yRegion2 = i
            break
    else:
        if yRegion2 == -1:
            yRegion2 = 4

    if yRegion1 < yRegion2:
        return -1
    elif yRegion2 < yRegion1:
        return 1
    elif x1 <= x2:
        return -1
    else:
        return 1


def dispImages(imgs):
    for img in imgs:
        cv2.imshow('Image', img)
        cv2.waitKey(0)
    else:
        cv2.destroyAllWindows()

