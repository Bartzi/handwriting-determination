import cv2
import numpy

from datasets.image_dataset import BaseImageDataset
from utils.bbox.bbox import AxisAlignedBBox
from utils.datatypes import Size


class PredictionDatasetMixin:

    def __init__(self, *args, **kwargs):
        kwargs['resize_after_load'] = False
        kwargs['transform_probability'] = 0

        image_size = kwargs.pop('image_size')
        if not isinstance(image_size, Size):
            image_size = Size(*image_size)

        self.max_size = kwargs.pop('max_size')
        self.image_size = image_size
        self.binarize_windows = kwargs.pop('binarize_windows', False)
        self.threshold_method = kwargs.pop('threshold_method', 'otsu')

        super().__init__()

    @staticmethod
    def get_num_windows_and_overlap(input_dimension, output_dimension):
        num_windows, rest = divmod(input_dimension, output_dimension)
        if rest != 0:
            overlap = (output_dimension - rest) // num_windows + 1
            num_windows += 1
        else:
            overlap = 0

        return num_windows, overlap

    def otsu_threshold(self, image):
        """
            Thresholds the image using Otsu's method.
            """
        blur = 3

        blurred = cv2.medianBlur(image, blur)
        _, threshold_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return threshold_image

    def adaptive_gaussian_threshold(self, image):
        """
        Thresholds the image using the Adaptive Gaussian method.
        """
        blurred = cv2.medianBlur(image, 3)
        threshold_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 41)

        return threshold_image

    def binarize_image(self, image):
        dtype = image.dtype
        image = image[0]
        image = (image * 255).astype('uint8')

        if self.threshold_method == 'otsu':
            thresholded = self.otsu_threshold(image)
        else:
            thresholded = self.adaptive_gaussian_threshold(image)

        thresholded = thresholded.astype(dtype)
        thresholded /= 255
        thresholded = numpy.stack([thresholded, thresholded, thresholded], axis=0)
        return thresholded

    def create_sliding_window(self, image):
        loaded_size = Size(*image.shape[-2:])
        windows_at_width, width_overlap = self.get_num_windows_and_overlap(loaded_size.width, self.image_size.width)
        windows_at_height, height_overlap = self.get_num_windows_and_overlap(loaded_size.height, self.image_size.height)

        bboxes = []
        crops = []
        for j in range(windows_at_height):
            for i in range(windows_at_width):
                bbox = AxisAlignedBBox(
                    left=i * self.image_size.width - i * width_overlap,
                    top=j * self.image_size.height - j * height_overlap,
                    right=(i + 1) * self.image_size.width - i * width_overlap,
                    bottom=(j + 1) * self.image_size.height - j * height_overlap,
                )
                bboxes.append(bbox)
                window = image[:, bbox.top:bbox.bottom, bbox.left:bbox.right]
                if self.binarize_windows:
                    window = self.binarize_image(window)
                crops.append(window)

        return numpy.stack(crops, axis=0), bboxes

    def resize_image(self, image):
        loaded_size = Size(*image.shape[-2:])

        if all(x < self.max_size for x in loaded_size) and loaded_size.width > self.image_size.width and loaded_size.height > self.image_size.height:
            return image

        if loaded_size.width > loaded_size.height:
            aspect_ratio = loaded_size.height / loaded_size.width
            new_size = Size(height=max(int(self.max_size * aspect_ratio), self.image_size.height), width=self.max_size)
        else:
            aspect_ratio = loaded_size.width / loaded_size.height
            new_size = Size(height=self.max_size, width=max(int(self.max_size * aspect_ratio), self.image_size.width))

        image = image.transpose(1, 2, 0)
        image = cv2.resize(image, (new_size.width, new_size.height), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)
        return image


class PredictionDataset(PredictionDatasetMixin, BaseImageDataset):

    def __init__(self, *args, **kwargs):
        self.binarize_input_image = kwargs.pop('binarize_image', False)
        super().__init__(*args, **kwargs)

    def get_example(self, i):
        data = super().get_example(i)
        image = self.resize_image(data['image'])

        if self.binarize_input_image:
            image = self.binarize_image(image)

        windows, bboxes = self.create_sliding_window(image)

        data['original_image'] = image
        data['image'] = windows
        data['boxes'] = bboxes
        return data
