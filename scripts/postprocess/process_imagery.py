import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MIN_RESOLUTION = 256  # API should provide 512 max thumbnail
FINAL_SIZE = (1024, 1024)

BAYER_4X4 = np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ]
) / 16.0 * 255


def _tile_threshold_map(threshold_map, width, height):
    """Repeat a small threshold map across an image-sized grid (wrap, like `map[y % h, x % w]`)."""
    reps_y = -(-height // threshold_map.shape[0])
    reps_x = -(-width // threshold_map.shape[1])
    return np.tile(threshold_map, (reps_y, reps_x))[:height, :width]


def dither_floyd_steinberg(gray):
    """Pillow's built-in error-diffusion dither (its default for 'L' -> '1')."""
    return gray.convert("1", dither=Image.Dither.FLOYDSTEINBERG)


def dither_halftone(gray):
    """Contrast-boosted Floyd-Steinberg, producing denser clustering of dots."""
    gray = ImageEnhance.Contrast(gray).enhance(2)
    return gray.convert("1", dither=Image.Dither.FLOYDSTEINBERG)


def dither_ordered(gray):
    """Ordered dithering via a tiled 4x4 Bayer matrix (vectorized, no per-pixel Python loop)."""
    pixels = np.array(gray)
    threshold = _tile_threshold_map(BAYER_4X4, pixels.shape[1], pixels.shape[0])
    bw = np.where(pixels > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(bw, mode="L").convert("1")


def dither_ordered_noise(gray, seed=42, blue_noise_size=64, blue_noise_scale=24):
    """Bayer matrix blended with blue noise for a less regular ordered-dither pattern."""
    pixels = np.array(gray)
    height, width = pixels.shape

    bayer = _tile_threshold_map(BAYER_4X4, width, height)

    rng = np.random.default_rng(seed)
    blue_noise = rng.random((blue_noise_size, blue_noise_size)) * 255
    noise = _tile_threshold_map(blue_noise, width, height) / 255.0 * blue_noise_scale

    bw = np.where(pixels > (bayer + noise), 255, 0).astype(np.uint8)
    return Image.fromarray(bw, mode="L").convert("1")


DITHER_ALGORITHMS = {
    "floyd-steinberg": dither_floyd_steinberg,
    "halftone": dither_halftone,
    "ordered": dither_ordered,
    "ordered-noise": dither_ordered_noise,
}


def process_image(
    image,
    output_path=None,
    aspect_ratio=None,
    final_size=FINAL_SIZE,
    min_resolution=MIN_RESOLUTION,
    algorithm="floyd-steinberg",
):
    """Dither `image` using the selected algorithm and make white pixels transparent."""
    if algorithm not in DITHER_ALGORITHMS:
        raise ValueError(f"Unknown dither algorithm '{algorithm}'. Choose from: {', '.join(DITHER_ALGORITHMS)}")

    width, height = image.size
    if width < min_resolution or height < min_resolution:
        return None

    # Resize (pre-dither) while maintaining aspect ratio if aspect_ratio is specified
    if aspect_ratio:
        side = min(width, height)
        image = image.resize((int(side * aspect_ratio), int(side)), Image.BILINEAR)

    gray = image.convert("L")
    image = DITHER_ALGORITHMS[algorithm](gray).convert("RGBA")

    # Convert white (and near-white) pixels to transparent
    data = np.array(image)
    white_areas = (data[:, :, 0] > 200) & (data[:, :, 1] > 200) & (data[:, :, 2] > 200)
    data[white_areas] = [255, 255, 255, 0]
    image = Image.fromarray(data)

    # Resize (post-dither) using a defined interpolation method - NEAREST, BILINEAR, BICUBIC
    image = image.resize(final_size, Image.NEAREST)

    # Crop the outer 2% after the final resize
    if aspect_ratio:
        width, height = image.size
        crop_pixels = int(min(width, height) * 0.02)  # 2% from each edge
        image = image.crop((crop_pixels, crop_pixels, width - crop_pixels, height - crop_pixels))

    if output_path is not None:
        image.save(output_path, optimize=True)

    return image


def find_image_files(directory):
    return sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def default_output_dir(algorithm):
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "output" / "processed" / f"{algorithm}_{timestamp}"


def main():
    parser = argparse.ArgumentParser(description="Batch-process a directory of images with a selectable dithering algorithm.")
    parser.add_argument("input_dir", help="Directory containing source images")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Directory to write processed images to (default: ../../output/processed/<algorithm>_<yyyymmddhhmm>)",
    )
    parser.add_argument(
        "--algorithm",
        choices=list(DITHER_ALGORITHMS),
        default="floyd-steinberg",
        help="Dithering algorithm to apply (default: floyd-steinberg)",
    )
    parser.add_argument("--aspect-ratio", type=float, default=None, help="Target width/height aspect ratio applied before dithering")
    parser.add_argument(
        "--final-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=FINAL_SIZE,
        help=f"Output image size in pixels (default: {FINAL_SIZE[0]} {FINAL_SIZE[1]})",
    )
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=MIN_RESOLUTION,
        help=f"Minimum source resolution required to process an image (default: {MIN_RESOLUTION})",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"Input directory '{input_dir}' does not exist.")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.algorithm)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = find_image_files(input_dir)
    if not image_files:
        print(f"No images found in '{input_dir}'.")
        return

    print(f"Found {len(image_files)} images in '{input_dir}'")
    print(f"Algorithm: {args.algorithm}")
    print(f"Writing processed images to '{output_dir}'")

    skipped = 0
    for path in tqdm(image_files, desc="Processing images"):
        with Image.open(path) as src:
            result = process_image(
                src,
                output_path=output_dir / f"{path.stem}.png",
                aspect_ratio=args.aspect_ratio,
                final_size=tuple(args.final_size),
                min_resolution=args.min_resolution,
                algorithm=args.algorithm,
            )
        if result is None:
            tqdm.write(f"Skipped '{path.name}': below minimum resolution ({args.min_resolution}px)")
            skipped += 1

    print(f"Processed {len(image_files) - skipped} images ({skipped} skipped) -> '{output_dir}'")


if __name__ == "__main__":
    main()
