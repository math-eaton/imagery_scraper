import os
import argparse
import imageio.v2 as imageio
from PIL import Image, ImageSequence
from tqdm import tqdm
import re
import numpy as np
import random

# note that based on optional params, the jitter effect only occurs if
# the frame disposal method is '3'

def resize_to_center(image, scale, bg_color, jitter=0.75):
    width, height = image.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    new_image = Image.new("RGBA", (width, height), bg_color)
    
    jitter_x = int((width - new_width) * jitter * (random.random() - 0.5))
    jitter_y = int((height - new_height) * jitter * (random.random() - 0.5))
    
    new_image.paste(resized_image, ((width - new_width) // 2 + jitter_x, (height - new_height) // 2 + jitter_y), resized_image)
    return new_image

def create_gif(image_files, gif_path, ping_pong=False, duration=250, disposal=2, bg_color=(0, 0, 0, 0)):
    images = [Image.open(x).convert("RGBA") for x in tqdm(image_files, desc="Reading images")]
    
    if ping_pong:
        images = images + images[-2:0:-1]
    
    if disposal == 3:
        scaled_images = []
        scale = 1.0
        for img in images:
            scaled_images.append(resize_to_center(img, scale, bg_color))
            scale *= random.uniform(0.96, 1.01)
        images = scaled_images
    
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=True,
        duration=duration,
        loop=0,
        disposal=disposal
    )

def create_gif_batch(image_files, gif_path, batch_size=500, ping_pong=False, duration=250, disposal=2, bg_color=(0, 0, 0, 0)):
    num_batches = len(image_files) // batch_size
    temp_gifs = []

    for i in tqdm(range(num_batches + 1), desc="Processing batches"):
        batch_images = image_files[i * batch_size: (i + 1) * batch_size]
        if not batch_images:
            continue
        images = [imageio.imread(x, mode='RGBA') for x in tqdm(batch_images, desc="Reading images", leave=False)]
        if ping_pong:
            images = images + images[-2:0:-1]
        if disposal == 3:
            scaled_images = []
            scale = 1.0
            for img in images:
                pil_img = Image.fromarray(img)
                scaled_images.append(resize_to_center(pil_img, scale, bg_color))
                scale *= random.uniform(0.96, 1.01)
            images = [np.array(img) for img in scaled_images]
        temp_gif_path = f"temp_{i}.gif"
        imageio.mimsave(temp_gif_path, images, 'GIF', duration=duration / 1000.0, loop=0, disposal=disposal)
        temp_gifs.append(temp_gif_path)

    final_gif_frames = []

    for temp_gif in tqdm(temp_gifs, desc="Concatenating GIFs"):
        with Image.open(temp_gif) as img:
            for frame in ImageSequence.Iterator(img):
                final_frame = frame.copy()
                final_gif_frames.append(final_frame)

    final_gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=final_gif_frames[1:],
        duration=duration,
        loop=0,
        disposal=disposal
    )

    for temp_gif in temp_gifs:
        os.remove(temp_gif)

def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)

def main():
    parser = argparse.ArgumentParser(description="Create unique GIFs from a directory of PNG images")
    parser.add_argument("input_folder", help="Path to the input folder containing PNG images")
    parser.add_argument("output_folder", help="Path to the output folder where GIFs will be saved")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for processing images (default: 500)")
    parser.add_argument("--ping-pong", action="store_true", help="Enable ping-pong looping for the GIFs")
    parser.add_argument("--duration", type=int, default=250, help="Duration of each frame in milliseconds (default: 250)")
    parser.add_argument("--disposal", type=int, default=2, help="Frame disposal method (default: 2)")
    parser.add_argument("--bg-color", type=str, default="#000000", help="Background color to replace transparent pixels (default: #000000)")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    batch_size = args.batch_size
    ping_pong = args.ping_pong
    duration = args.duration
    disposal = args.disposal
    bg_color = hex_to_rgba(args.bg_color)

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Batch size: {batch_size}")
    print(f"Ping-pong mode: {'enabled' if ping_pong else 'disabled'}")
    print(f"Frame duration: {duration} ms")
    print(f"Frame disposal: {disposal}")
    print(f"Background color: {args.bg_color}")

    if not os.path.exists(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder '{output_folder}'")

    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    print(f"Found {len(images)} images in the input folder")

    # Group images by the prefix before the first underscore
    grouped_images = {}
    for img in images:
        match = re.match(r"([^_]+)_(\d+).png", img)
        if match:
            prefix = match.group(1)
            index = int(match.group(2))
            if prefix not in grouped_images:
                grouped_images[prefix] = []
            grouped_images[prefix].append((index, os.path.join(input_folder, img)))

    print(f"Grouped images into {len(grouped_images)} categories")

    for prefix, files in grouped_images.items():
        print(f"Processing group '{prefix}' with {len(files)} images")
        files.sort()  # Sort by the numerical index
        sorted_files = [f[1] for f in files]
        output_gif = os.path.join(output_folder, f"{prefix}.gif")
        create_gif_batch(sorted_files, output_gif, batch_size=batch_size, ping_pong=ping_pong, duration=duration, disposal=disposal, bg_color=bg_color)
        print(f"Created GIF '{output_gif}'")

if __name__ == "__main__":
    main()
