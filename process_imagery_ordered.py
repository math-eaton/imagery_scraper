from PIL import Image
import numpy as np
import os

# Process the image using Floyd-Steinberg error diffusion
def process_image(image, output_path, aspect_ratio=None, final_size=(720, 720)):
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
    
    # Apply ordered dithering using Bayer matrix
    image = image.convert('1', dither=Image.Dither.NONE)
    threshold_map = np.array([[0, 8, 2, 10],
                              [12, 4, 14, 6],
                              [3, 11, 1, 9],
                              [15, 7, 13, 5]])
    threshold_map = threshold_map / 16.0 * 255

    pixels = np.array(image)
    for y in range(image.height):
        for x in range(image.width):
            i = x % 4
            j = y % 4
            pixels[y, x] = 255 if pixels[y, x] > threshold_map[j, i] else 0
    image = Image.fromarray(pixels)

    image = image.convert('RGB')
    image = image.convert('RGBA')

    data = np.array(image)
    red, green, blue, alpha = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    white_areas = (red == 255) & (green == 255) & (blue == 255)
    data[white_areas] = [255, 255, 255, 0]
    image = Image.fromarray(data)

    image = image.resize(final_size, Image.NEAREST)
    print(f"rescaling to {final_size} pixels")

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
