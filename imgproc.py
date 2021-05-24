def allocate_borders(source_image, image_with_allocated_borders, start_pixel, end_pixel):
    for i in range(1, len(source_image) - 1):
        for j in range(start_pixel, end_pixel):
            if int(source_image[i][j]) != 0:
                squares_of_differences_sum = get_squares_of_differences_sum(source_image, [i, j])
                image_with_allocated_borders[i][j] = squares_of_differences_sum


def get_squares_of_differences_sum(source_image, current_pixel_indexes):
    i, j = current_pixel_indexes
    center_pixel = int(source_image[i][j])  # vector[3]
    right_pixel = int(source_image[i][j + 1])  # vector[2]
    bottom_pixel = int(source_image[i + 1][j])  # vector[0]
    bottom_right_corner_pixel = int(source_image[i + 1][j + 1])  # vector[1]
    squares_of_differences_sum = abs(bottom_right_corner_pixel - center_pixel) + abs(bottom_pixel - right_pixel)
    return squares_of_differences_sum


def scale_image(source_image, scaled_image, scale_size, start_pixel, end_pixel):
    scaled_image_size = get_scaled_image_size(source_image, scale_size)
    scale_factors = get_scale_factors(scaled_image_size, {"width": len(source_image), "height": len(source_image)})
    for i in range(scaled_image_size["width"] - 1):
        for j in range(start_pixel, end_pixel - 1):
            scaled_image[i + 1, j + 1] = source_image[1 + int(i / scale_factors["width"]),
                                                      1 + int(j / scale_factors["height"])]


def get_scale_factors(scaled_image_size, source_image_size):
    width_scale = scaled_image_size["width"] / (source_image_size["width"] - 1)
    height_scale = scaled_image_size["height"] / (source_image_size["height"] - 1)
    scale_factors = {"width": width_scale, "height": height_scale}
    return scale_factors


def get_scaled_image_size(source_image, scale_size):
    width, height = source_image.shape[:2]
    scaled_width = int(width * scale_size)
    scaled_height = int(height * scale_size)
    scaled_image_size = {'width': scaled_width, 'height': scaled_height}
    return scaled_image_size
