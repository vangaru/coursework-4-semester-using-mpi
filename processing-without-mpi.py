from imgpath import ImagePath
from imgproc import *
from datetime import datetime
from copy import deepcopy
import numpy as np
from cv2 import *

IMG_SIZE = 5000
SCALE_SIZE = 2

if __name__ == '__main__':
    source_image = imread(ImagePath.SRC.value, IMREAD_GRAYSCALE)

    print("BORDER ALLOCATION WITHOUT MULTIPROCESSING")

    image_with_allocated_borders = deepcopy(source_image)

    timer_start = datetime.now()

    allocate_borders(source_image, image_with_allocated_borders, 0, IMG_SIZE - 1)

    timer_end = datetime.now()
    timer_result = timer_end - timer_start

    imwrite(ImagePath.ALLOCATED_BORDERS.value, image_with_allocated_borders)

    print("Image with allocated borders: {}".format(ImagePath.ALLOCATED_BORDERS.value))
    print("Time: {}".format(timer_result))

    print("\nSCALING IMAGE WITHOUT MULTIPROCESSING")

    scaled_image_size = get_scaled_image_size(source_image, SCALE_SIZE)
    scaled_image = np.zeros([scaled_image_size["height"], scaled_image_size["width"]])

    timer_start = datetime.now()

    scale_image(source_image, scaled_image, SCALE_SIZE, 0, scaled_image_size["width"])

    timer_end = datetime.now()
    timer_result = timer_end - timer_start

    imwrite(ImagePath.SCALED.value, scaled_image)

    print("Scaled image: {}".format(ImagePath.SCALED.value))
    print("Scale size: {}".format(SCALE_SIZE))
    print("Time: {}".format(timer_result))
