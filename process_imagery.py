from PIL import Image
import numpy as np

# Process the image using Floyd-Steinberg error diffusion
def process_image(image_path, output_path):
    # Open the image
    image = Image.open(image_path)
    
    # Check the input image resolution
    min_resolution = 400  # Set minimum resolution. API should provide 512 max thumbnail
    width, height = image.size
    if width < min_resolution or height < min_resolution:
        print(f"Image for application_id {os.path.basename(image_path).split('.')[0]} was not processed due to low resolution.")
        return None
    
    # Resize the image (pre-dither) while maintaining aspect ratio
    size = (2048, 2048)  # Set your desired size here
    image.thumbnail(size, Image.BILINEAR)
    
    # Crop the image to desired aspect ratio
    # width, height = image.size
    # new_size = min(width, height)

    # left = (width - new_size)/2
    # top = (height - new_size)/2
    # right = (width + new_size)/2
    # bottom = (height + new_size)/2

    # # Calculate and print the aspect ratio
    # aspect_ratio = new_size / new_size  # e.g. square is 1.0

    # image = image.crop((left, top, right, bottom))
    # print(f"cropping to aspect ratio {aspect_ratio}")

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

    # Crop the outer 2%
    width, height = image.size
    left = width * 0.02
    top = height * 0.02
    right = width * 0.98
    bottom = height * 0.98
    image = image.crop((left, top, right, bottom))

    # Resize the image (post-dither) using nearest neighbor
    size = (1200, 1200)  # Set your desired size here
    # size = (1200, 900)  # Size for video
    image = image.resize(size, Image.NEAREST)
    print("rescaling...")

    # Save the processed image
    image.save(output_path)

    return image