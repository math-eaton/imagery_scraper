from PIL import Image
import numpy as np
import os

# Process the image using Floyd-Steinberg error diffusion
def process_image(image, output_path, aspect_ratio=None, final_size=(800, 800)):
    # Check the input image resolution
    min_resolution = 400  # Set minimum resolution. API should provide 512 max thumbnail
    width, height = image.size
    if width < min_resolution or height < min_resolution:
        print(f"Image for application_id {os.path.basename(image_path).split('.')[0]} was not processed due to low resolution.")
        return None

    # Resize the image (pre-dither) while maintaining aspect ratio if aspect_ratio is specified
    if aspect_ratio:
        new_width = int(min(width, height) * aspect_ratio)
        new_height = int(min(width, height))
        image = image.resize((new_width, new_height), Image.BILINEAR)

    # Convert the image to grayscale
    image = image.convert('L')

    # Dither the image
    image = image.convert('1')
    print("dithering...")

    # Convert the image back to RGB
    image = image.convert('RGB')

    # Make sure the image has an alpha channel
    image = image.convert('RGBA')

    # Convert white (also shades of whites) pixels to transparent
    data = np.array(image)
    red, green, blue, alpha = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    white_areas = (red > 200) & (green > 200) & (blue > 200)
    data[white_areas] = [255, 255, 255, 0]
    image = Image.fromarray(data)

    # Resize the image (post-dither) using a defined interpolation method - NEAREST, BILINEAR, BICUBIC
    image = image.resize(final_size, Image.NEAREST)
    print(f"rescaling to {final_size} pixels")

    # Crop the outer 2% after the final resize
    if aspect_ratio:
        width, height = image.size
        crop_percentage = 2  # Crop 2% from each edge
        crop_pixels = int(min(width, height) * (crop_percentage / 100))

        left = crop_pixels
        top = crop_pixels
        right = width - crop_pixels
        bottom = height - crop_pixels

        image = image.crop((left, top, right, bottom))

    # Save the processed image
    image.save(output_path, optimize=True)

    return image