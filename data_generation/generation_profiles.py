import os
from image_transformations import otsu_threshold, lighter_otsu_threshold, adaptive_gaussian_threshold, \
    etched_lines

IMAGE_FORMATS = ('.png', '.JPG', 'jpg', '.JPEG', '.jpeg')

BASE_DIR = "[path to base dir]"

WORD_PATH = os.path.join(BASE_DIR, 'words')
PHOTOS_PATH = os.path.join(BASE_DIR, 'backgrounds')
PRINT_FONTS_PATH = os.path.join(BASE_DIR, 'fonts')
PRINT_DOCUMENT_PATH = os.path.join(BASE_DIR, 'documents')
HANDWRITTEN_DOCUMENT_PATH = os.path.join(BASE_DIR, 'handwritten_documents')
ARTIFACT_PATH = os.path.join(BASE_DIR, 'artifacts')


def load_image_paths(base_path):
    """
    Finds all images in the given base path and returns the paths to them
    """
    return [os.path.join(root, name)
            for root, dirs, files in os.walk(base_path)
            for name in files if name.endswith(IMAGE_FORMATS)]

# load the paths of all available images
word_image_paths = load_image_paths(WORD_PATH)
photo_image_paths = load_image_paths(PHOTOS_PATH)
fonts_image_paths = load_image_paths(PRINT_FONTS_PATH)
document_image_paths = load_image_paths(PRINT_DOCUMENT_PATH)
artifact_image_paths = load_image_paths(ARTIFACT_PATH)
handwritten_document_image_paths = load_image_paths(HANDWRITTEN_DOCUMENT_PATH)

# define the profiles that will later on be used to generate fragments with certain traits,
# one profile defines one trait

# handwritten words
WRITING_CANVAS_PROFILE = {
    'image_paths': word_image_paths,                            # images which will be used for generating this trait
    'scale_range': (0.25, 1),                                   # how strong will the images be scaled?
    'rotation_range': (0, 360),                                 # how far will the images be rotated?
    'preprocessing_method': otsu_threshold,                     # what preprocessing should be applied to images
    'dilation_erosion': [(-3, 1), (3, 1), None, None, None],    # what dilation/erosion parameters should be used?
                                                                # None means nothing is applied, negative means Erosion
    'min_black_share': 0,                                       # what is the minimum amount of black required?
    'cutout': False                                             # cutout from the images or paste the whole image?
}

# graphical elements after preprocessing, i.e. organic shapes
GRAPHICAL_CANVAS_PROFILE = {
    'image_paths': photo_image_paths,
    'scale_range': (2, 10),
    'rotation_range': (0, 360),
    'preprocessing_method': adaptive_gaussian_threshold,
    'dilation_erosion': [(3, 1), None],
    'min_black_share': 0,
    'cutout': True
}

# shapes that appear after preprocessing a drawing
DRAWING_CANVAS_PROFILE = {
    'image_paths': photo_image_paths,
    'scale_range': (2, 10),
    'rotation_range': (0, 360),
    'preprocessing_method': etched_lines,
    'dilation_erosion': [(-3, 1), (3, 1), (5, 1), None],
    'min_black_share': 0.05,
    'cutout': True
}

# words in different fonts
FONT_CANVAS_PROFILE = {
    'image_paths': fonts_image_paths,
    'scale_range': (0.4, 1),
    'rotation_range': (0, 360),
    'preprocessing_method': otsu_threshold,
    'dilation_erosion': [(-3, 1), (3, 1), (5, 1), None],
    'min_black_share': 0.05,
    'cutout': False
}

# multiple lines of printed text
DOCUMENT_CANVAS_PROFILE = {
    'image_paths': document_image_paths,
    'scale_range': (5, 50),
    'rotation_range': (0, 360),
    'preprocessing_method': otsu_threshold,
    'dilation_erosion': [(-3, 1), (3, 1), None, None],
    'min_black_share': 0,
    'cutout': True
}

# artifacts that appear when preprocessing the original data
ARTIFACT_CANVAS_PROFILE = {
    'image_paths': artifact_image_paths,
    'scale_range': (1, 5),
    'rotation_range': (0, 15),
    'preprocessing_method': otsu_threshold,
    'dilation_erosion': [None, (-3, 1)],
    'min_black_share': 0,
    'cutout': False
}

# handwritten lines of text
HANDWRITTEN_DOCUMENT_PROFILE = {
    'image_paths': handwritten_document_image_paths,
    'scale_range': (5, 50),
    'rotation_range': (0, 360),
    'preprocessing_method': adaptive_gaussian_threshold,
    'dilation_erosion': [(3, 1), (-3, 1), None, None],
    'min_black_share': 0.05,
    'cutout': True
}

