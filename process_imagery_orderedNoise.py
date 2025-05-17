from PIL import Image
import numpy as np
import os

# Generate or load blue noise
def generate_blue_noise(size):
    np.random.seed(42)
    return np.random.rand(size, size) * 255

# Process the image using a hybrid of Bayer matrix and blue noise
def process_image(image, output_path, aspect_ratio=None, final_size=(1920, 1920)):
    min_resolution = 400
    width, height = image.size
    if width < min_resolution or height < min_resolution:
        print(f"Image for application_id {os.path.basename(image_path).split('.')[0]} was not processed due to low resolution.")
        return None

    if aspect_ratio:
        new_width = int(min(width, height) * aspect_ratio)
        new_height = int(min(width, height))
        image = image.resize((new_width, new_height), Image.BILINEAR)

    image = image.convert('L')
    
    # Bayer threshold map (4x4)
    bayer_matrix = np.array([[0, 8, 2, 10],
                             [12, 4, 14, 6],
                             [3, 11, 1, 9],
                             [15, 7, 13, 5]])
    bayer_matrix = bayer_matrix / 16.0 * 255

    # Load or generate blue noise
    blue_noise_size = 64  # Ensure the size is larger than the image dimensions
    blue_noise = generate_blue_noise(blue_noise_size)

    # Combine Bayer matrix with blue noise (scaled appropriately)
    def get_threshold(x, y):
        i = x % 4
        j = y % 4
        blue_noise_offset = blue_noise[y % blue_noise_size, x % blue_noise_size] / 255.0
        return bayer_matrix[j, i] + blue_noise_offset * 24  # Adjust scale of blue noise

    # Apply dithering with blue noise offsets
    pixels = np.array(image)
    for y in range(image.height):
        for x in range(image.width):
            threshold = get_threshold(x, y)
            pixels[y, x] = 255 if pixels[y, x] > threshold else 0
    
    image = Image.fromarray(pixels)

    # Convert image to RGBA and make white areas transparent
    image = image.convert('RGB').convert('RGBA')
    data = np.array(image)
    red, green, blue, alpha = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    white_areas = (red == 255) & (green == 255) & (blue == 255)
    data[white_areas] = [255, 255, 255, 0]
    image = Image.fromarray(data)

    image = image.resize(final_size, Image.NEAREST)
    print(f"Rescaling to {final_size} pixels")

    if aspect_ratio:
        width, height = image.size
        crop_percentage = 2
        crop_pixels = int(min(width, height) * (crop_percentage / 100))

        left = crop_pixels
        top = crop_pixels
        right = width - crop_pixels
        bottom = height - crop_pixels

        image = image.crop((left, top, right, bottom))

    image.save(output_path, optimize=True)
    return image
