def add_impulsive_noise(image, noise_factor=0.1):
    """
    Add impulsive noise to an image.
    Parameters:
        - image: The image (2D or 3D array) to add noise to.
        - noise_factor: Proportion of image pixels to modify.
    """
    np.random.seed(42)

    noisy_image = image.copy()

    #set some pixels to 1
    white_pixels = np.random.rand(*image.shape) < (noise_factor / 2)
    noisy_image[white_pixels] = 1

    # set some pixels to 0
    black_pixels = np.random.rand(*image.shape) < (noise_factor / 2)

    noisy_image[black_pixels] = 0

    return noisy_image
