from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image

from config import DEFAULT_GRAYSCALE, DEFAULT_IMAGE_SIZE, NORMALIZATION_LABEL


def image_to_vector(
    image_file: BinaryIO | bytes,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    grayscale: bool = DEFAULT_GRAYSCALE,
) -> list[float]:
    raw_image = image_file if isinstance(image_file, bytes) else image_file.read()
    image = Image.open(BytesIO(raw_image))

    # Format modèle
    if grayscale:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize(image_size, Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32).reshape(-1)

    # Pixels [-1, 1]
    return ((values / 127.5) - 1.0).tolist()


def preprocessing_metadata(
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    grayscale: bool = DEFAULT_GRAYSCALE,
) -> dict[str, object]:
    width, height = image_size
    channel_count = 1 if grayscale else 3

    # Infos API
    return {
        "width": width,
        "height": height,
        "grayscale": grayscale,
        "channels": channel_count,
        "input_dim": width * height * channel_count,
        "normalization": NORMALIZATION_LABEL,
        "resampling": "bilinear",
    }
