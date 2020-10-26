import math
import random
from random import randint
import cv2
import numpy as np
from generation_profiles import GRAPHICAL_CANVAS_PROFILE, DRAWING_CANVAS_PROFILE, \
    ARTIFACT_CANVAS_PROFILE, FONT_CANVAS_PROFILE, DOCUMENT_CANVAS_PROFILE, WRITING_CANVAS_PROFILE, \
    HANDWRITTEN_DOCUMENT_PROFILE
from random import choices
from image_transformations import random_scale, random_rotation, paste_at_random_location, \
    paste_cutout_at_random_location, random_dilation_erosion

# max amount of cached images, if the cache is full all new images will not be cached
MAX_CACHE_SIZE = 1000000

MIN_IMAGE_SIZE = 20 * 20

# when evaluating positive fragments (with handwriting) the handwriting will account for at least
# MAX_ACCEPTED_SHARE of all black pixels, this prevents that images are generated in which
# no handwriting is actually visible
MAX_ACCEPTED_SHARE = 0.075

# determines how much percent of the positive handwriting images should also contain background elements
COMBINED_HANDWRITING_RATIO = 0.65

image_cache = {}


def empty_canvas(dimensions):
    """
    Generates a white (empty) canvas, i.e. image that will be used for the fragment generation
    """
    canvas = np.ones((dimensions[1], dimensions[0]), np.uint8) * 255
    return canvas


def generate_canvas(profile, dimensions):
    """
    Generates a canvas based on the given profile
    """
    image_paths = profile['image_paths']
    min_black_share = profile['min_black_share']

    current_black_share = 0
    canvas = None

    while canvas is None or current_black_share < min_black_share:
        random_image = get_random_image(image_paths)
        canvas = apply_profile_transformations(profile, random_image, dimensions)
        if min_black_share > 0:
            current_black_share = black_share_in_canvas(canvas)

    return canvas


def apply_profile_transformations(profile, image, dimensions):
    """
    Applies image transformations to the given image according to the given profile and image dimensions
    """
    preprocessing_method = profile['preprocessing_method']
    scale_range = profile['scale_range']
    rotation_range = profile['rotation_range']
    dilation_erosion_values = profile['dilation_erosion']
    cutout = profile['cutout']

    cutout_dimension = math.floor(math.sqrt(dimensions[0] ** 2 + dimensions[1] ** 2) * 2)

    canvas = empty_canvas(dimensions)
    image = random_scale(image, canvas, scale_range)

    # if a cutout is requested, we select the cutout-region before rotating to prevent that whitespace is selected
    # that only exists because of the rotation
    if cutout and image.shape[0] > cutout_dimension and image.shape[1] > cutout_dimension:
        # only do a cutout if the source image is large enough
        cutout_canvas = np.ones((cutout_dimension, cutout_dimension), np.uint8) * 255
        image = paste_cutout_at_random_location(image, cutout_canvas)

    image = random_rotation(image, angle_range=rotation_range)

    canvas = paste_at_random_location(image, canvas)
    canvas = preprocessing_method(canvas)
    canvas = random_dilation_erosion(canvas, dilation_erosion_values)

    return canvas


def get_random_image(image_paths):
    """
    Randomly picks a path from the given image paths and loads the corresponding image. A new image is randomly selected
    if the image cannot be loaded.
    """
    random_image = None  # reattempt to find an image if loading was not successful
    while random_image is None:
        random_image_path = image_paths[randint(0, len(image_paths) - 1)]
        if random_image_path in image_cache:
            random_image = image_cache[random_image_path]
        elif len(image_cache) == MAX_CACHE_SIZE:
            random_image = cv2.imread(random_image_path)
        else:
            random_image = cv2.imread(random_image_path)
            image_cache[random_image_path] = random_image

        # skip images that are very small, happens e.g. for punctuation in the IAM dataset
        if random_image is not None and random_image.shape[0] * random_image.shape[1] < MIN_IMAGE_SIZE:
            random_image = None
            image_cache[random_image_path] = random_image

    random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2GRAY)
    return random_image


def evaluate_share_with_background(word_canvas, background_canvas):
    """
    Evaluates the precentage of black pixels of the combined word and background canvas that belong to the word canvas.
    """
    fragment_overlap = cv2.bitwise_or(background_canvas, word_canvas)

    word_black_amount = np.count_nonzero(word_canvas == 0)

    if word_black_amount == 0:
        return math.inf

    share_with_background = np.count_nonzero(fragment_overlap == 0) / word_black_amount
    return share_with_background


def black_share_in_canvas(canvas):
    """
    Calculates the precentage of black pixels in the given canvas.
    """
    return np.count_nonzero(canvas == 0) / canvas.size


def generate_fragment(traits, dimension):
    """
    Generates a fragment with the the given traits and dimensions.
    """
    while True:
        background_canvas = empty_canvas(dimension)

        if 'GRAPHICAL' in traits:
            graphical_canvas = generate_canvas(GRAPHICAL_CANVAS_PROFILE, dimension)
            background_canvas = cv2.bitwise_and(background_canvas, graphical_canvas)

        if 'DRAWING' in traits:
            drawing_canvas = generate_canvas(DRAWING_CANVAS_PROFILE, dimension)
            background_canvas = cv2.bitwise_and(background_canvas, drawing_canvas)

        if 'ARTIFACT' in traits:
            artifact_canvas = generate_canvas(ARTIFACT_CANVAS_PROFILE, dimension)
            background_canvas = cv2.bitwise_and(background_canvas, artifact_canvas)

        if 'FONT' in traits:
            fonts_canvas = generate_canvas(FONT_CANVAS_PROFILE, dimension)
            background_canvas = cv2.bitwise_and(background_canvas, fonts_canvas)

        # the document canvas should not account in evaluate_share_with_background of the writing canvas (see below)
        # therefore we generate it on a separate document_canvas rather than the background_canvas
        document_canvas = empty_canvas(dimension)
        if 'DOCUMENT' in traits:
            document_canvas = generate_canvas(DOCUMENT_CANVAS_PROFILE, dimension)

        if 'WRITING' in traits:
            writing_canvas = generate_canvas(WRITING_CANVAS_PROFILE, dimension)

            # check how much of the black actually belongs to the writing
            share_with_background = evaluate_share_with_background(writing_canvas, background_canvas)
            if share_with_background <= MAX_ACCEPTED_SHARE:
                # add the document canvas to the backround canvas after evaluation
                background_canvas = cv2.bitwise_and(background_canvas, document_canvas)
                fragment = cv2.bitwise_and(background_canvas, writing_canvas)
                return fragment

        elif 'HANDWRITTEN_DOCUMENT' in traits:
            handwritten_doc_canvas = generate_canvas(HANDWRITTEN_DOCUMENT_PROFILE, dimension)
            fragment = cv2.bitwise_and(background_canvas, handwritten_doc_canvas)
            return fragment

        else:
            # if the fragment does not contain any positive features (handwriting or handwritten documents)
            # just return it
            background_canvas = cv2.bitwise_and(background_canvas, document_canvas)
            return background_canvas


def positive_trait():
    """
    Returns a list of traits for a positive fragment, i.e. either contains a handwritten word or handwritten document
    """
    if random.uniform(0, 1) <= COMBINED_HANDWRITING_RATIO:
        traits = np.random.choice(BACKGROUND_TRAITS, number_of_background_traits(),
                                  p=POSITIVE_PROBABILITIES, replace=False)
        traits = list(traits)
    else:
        traits = []

    if random.randint(0, 1):
        traits.append('WRITING')
    else:
        traits.append('HANDWRITTEN_DOCUMENT')
        if 'GRAPHICAL' in traits: traits.remove('GRAPHICAL')
        if 'DOCUMENT' in traits: traits.remove('DOCUMENT')
    return traits


def negative_trait():
    """
    Returns a list of traits for a negative fragment, i.e. does not contain any handwritten elements
    """
    traits = np.random.choice(BACKGROUND_TRAITS, number_of_background_traits(),
                              p=NEGATIVE_PROBABILITIES, replace=False)
    return list(traits)


def number_of_background_traits():
    """
    Returns a number that determines how many traits a negative fragment should have
    """
    number_choices = [1, 2, 3, 4, 5, 6]
    weights = [0.7, 0.15, 0.075, 0.025, 0.025, 0.025]
    negative_traits = choices(number_choices, weights)
    # negative_traits = random.randint(1, int(len(BACKGROUND_TRAITS) / 2))
    return negative_traits


BACKGROUND_TRAITS = ['GRAPHICAL', 'DRAWING', 'FONT', 'DOCUMENT', 'ARTIFACT', '']
POSITIVE_TRAITS = ['WRITING', 'HANDWRITTEN_DOCUMENT']

# determines the probability of each trait to appear in a fragment (different probabilities for positive and negative
# fragments)
NEGATIVE_PROBABILITIES = [0.05, 0.3, 0.05, 0.35, 0.2, 0.05]
POSITIVE_PROBABILITIES = [0.05, 0.25, 0.05, 0.05, 0.1, 0.5]
