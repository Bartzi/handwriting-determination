import chainer

from PIL import Image


def aspect_ratio_preserving_resize(the_image, image_size):
    ratio = min(image_size.width / the_image.width, image_size.height / the_image.height)
    the_image = the_image.resize((int(ratio * the_image.width), int(ratio * the_image.height)), Image.LANCZOS)

    # paste resized image into blank image to have image of correct input size
    blank_image = Image.new(the_image.mode, (image_size.width, image_size.height))
    # determine center of image and paste new image to be in the center!
    paste_start_x = image_size.width // 2 - the_image.width // 2
    paste_start_y = image_size.height // 2 - the_image.height // 2
    blank_image.paste(the_image, (paste_start_x, paste_start_y))

    return blank_image


def prepare_image(the_image, image_size, xp, keep_aspect_ratio, normalize=False, do_resize=True):
    if do_resize:
        if keep_aspect_ratio:
            the_image = aspect_ratio_preserving_resize(the_image, image_size)
        else:
            the_image = the_image.resize((image_size.width, image_size.height), Image.LANCZOS)

    image = xp.array(the_image, chainer.get_dtype())

    if normalize:
        image -= image.min()
        image /= max(image.max(), 1)
    else:
        image /= 255
    # image = image * 2 - 1
    return xp.transpose(image, (2, 0, 1))
