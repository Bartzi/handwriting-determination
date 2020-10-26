import math
import random
import cv2
import numpy as np


def paste_image(image, canvas, position):
    """
    Pastes the given image on the canvas at the given position. The position denotes the center of the pasted image.
    """
    x_offset = int(math.floor(position[0] - (image.shape[1] / 2)))
    y_offset = int(math.floor(position[1] - (image.shape[0] / 2)))

    pasted_part_start_x = max(0, x_offset * -1)
    pasted_part_start_y = max(0, y_offset * -1)
    pasted_part_end_x = min(image.shape[1], canvas.shape[1] - x_offset)
    pasted_part_end_y = min(image.shape[0], canvas.shape[0] - y_offset)

    pasted_part = image[pasted_part_start_y:pasted_part_end_y, pasted_part_start_x:pasted_part_end_x]

    b_start_x = max(0, x_offset)
    b_start_y = max(0, y_offset)

    canvas[b_start_y:b_start_y+pasted_part.shape[0], b_start_x:b_start_x+pasted_part.shape[1]] = pasted_part
    return canvas


def rotate_image(img, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = img.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_image = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), borderValue=(255,255,255))
    return rotated_image


def random_scale(image, canvas, scale_range=(0.75, 2)):
    """
    Scales the given image randomly within the given scale range.
    The scale factors are given according to the size of the given canvas.
    """
    scale_factor = random.uniform(scale_range[0], scale_range[1])
    current_scale = image.shape[0] / canvas.shape[0]
    new_scale = scale_factor / current_scale
    return cv2.resize(image, (0, 0), fx=new_scale, fy=new_scale)


def random_rotation(image, angle_range=(-90, 90)):
    """
    Rotates the given image randomly according to the given range. The angles should be given in degrees.
    """
    rotation_angle = random.randint(angle_range[0], angle_range[1])
    return rotate_image(image, rotation_angle % 360)


def paste_at_random_location(image, canvas):
    """
    Pastes the given image at a random location on the canvas.
    """
    random_location = (random.randint(0, canvas.shape[1]), random.randint(0, canvas.shape[0]))
    return paste_image(image, canvas, random_location)


def paste_cutout_at_random_location(image, canvas):
    """
    Selects a random cutout of the given image with the dimensions of the canvas and pastes it on the canvas.
    """
    half_shape = tuple(int(element / 2) for element in image.shape)
    random_location = (random.randint(canvas.shape[1] - half_shape[1], half_shape[1]),
                       random.randint(canvas.shape[0] - half_shape[0], half_shape[0]))
    return paste_image(image, canvas, random_location)


def random_threshold(image, threshold_range=(0, 125)):
    """
    Applies a random threshold to the image according to the given threshold range.
    """
    blur = 3

    blurred = cv2.medianBlur(image, blur)
    threshold = random.randint(threshold_range[0], threshold_range[1])
    _, threshold_image = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    return threshold_image


def random_dilation_erosion(image, values):
    """
    Selects one value of the given parameter values randomly and applies a dilation or erosion
    (if negative values were passed)with these parameters to the image.
    """
    value = random.choice(values)

    if value is None:
        return image

    kernel = np.ones((abs(value[0]), abs(value[0])), np.uint8)  # set kernel as nxn matrix from numpy

    if value[0] >= 0:
        return cv2.dilate(image, kernel, iterations=value[1])
    else:
        return cv2.erode(image, kernel, iterations=value[1])


def otsu_threshold(image):
    """
    Thresholds the image using Otsu's method.
    """
    blur = 3

    blurred = cv2.medianBlur(image, blur)
    _, threshold_image = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return threshold_image


def lighter_otsu_threshold(image):
    """
    Determines a threshold for the image using Otsu's method and decreases the threshold using a random factor.
    """
    blur = 3

    blurred = cv2.medianBlur(image, blur)
    otsu_threshold_level, _ = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    threshold_factor = random.uniform(0.4, 0.65)
    lighter_threshold_level = otsu_threshold_level * threshold_factor

    _, threshold_image = cv2.threshold(blurred, lighter_threshold_level, 255, cv2.THRESH_BINARY)

    return threshold_image


def adaptive_gaussian_threshold(image):
    """
    Thresholds the image using the Adaptive Gaussian method.
    """
    blurred = cv2.medianBlur(image, 3)

    threshold_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                           cv2.THRESH_BINARY, 61, 41)
    kernel = np.ones((3, 3), np.uint8)  # set kernel as 3x3 matrix from numpy
    erosion_image = cv2.erode(threshold_image, kernel, iterations=1)

    return erosion_image


def etched_lines(image):
    """
    Filters the given image to a representation that is similar to a drawing being preprocessed with an Adaptive Gaussian
    Threshold
    """
    block_size = 61
    c = 41
    blur = 7
    max_value = 255

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(image, (21, 21), 0, 0)
    img_blend = cv2.divide(image, img_blur, scale=256)

    blurred = cv2.medianBlur(img_blend, blur)
    threshold_image = cv2.adaptiveThreshold(blurred, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                            cv2.THRESH_BINARY, block_size, c)
    return threshold_image
