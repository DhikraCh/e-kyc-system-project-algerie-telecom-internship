import cv2
import numpy as np
from matplotlib import pyplot as plt


def process_image(image_path):

    image = cv2.imread(image_path)

    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    retval, dst = cv2.threshold(
        img_gray,
        150,
        255,
        cv2.THRESH_BINARY
    )

    # Save threshold image
    cv2.imwrite("image_thres1.jpg", dst)

    # Find contours
    contours, hierarchy = cv2.findContours(
        dst,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_copy = image.copy()

    cv2.drawContours(
        image=image_copy,
        contours=contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    # Save contours image
    cv2.imwrite(
        "contours_none_image1.jpg",
        image_copy
    )

    # Find a suitable ID rectangle
    c = None

    for contour in contours:

        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)

        ratio = max(w, h) / min(w, h)

        if area > 500 and 1.3 < ratio < 1.9:

            c = contour

            break

    # No suitable rectangle found
    if c is None:

        print("ID card rectangle not detected.")

        return None

    mask = np.zeros_like(img_gray)

    cv2.drawContours(
        mask,
        [c],
        -1,
        255,
        -1
    )

    extracted_crop = cv2.bitwise_and(
        image_copy,
        image_copy,
        mask=mask
    )

    x, y, w, h = cv2.boundingRect(c)

    cropped_roi = extracted_crop[
        y:y+h,
        x:x+w
    ]

    # Print contour coordinates
    for i, contour in enumerate(contours):

        x, y, w, h = cv2.boundingRect(contour)

        print(
            i,
            "x:", x,
            "y:", y,
            "w:", w,
            "h:", h
        )

    return cropped_roi