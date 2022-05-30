from pathlib import Path
from hooknet.configuration.config import create_hooknet
from wholeslidedata.iterators import create_batch_iterator
from wholeslidedata.source.configuration.config import insert_paths_into_config
from hooknet.inference.apply import _execute_inference_single
from hooknet.inference.writing import MaskType
import click
import numpy as np

from .constants import (
    HOOKNET_CONFIG,
    SEGMENTATION_CONFIG,
)


@click.command()
@click.option("--image_path", type=Path, required=True)
@click.option("--mask_path", type=Path, required=True)
@click.option("--output_folder", type=Path, required=True)
@click.option("--tmp_folder", type=Path, required=True)
@click.option("--name", type=Path, required=True)
def run_segmentation(image_path, mask_path, output_folder, tmp_folder, name):
    files = [
        {"name": name, "type": MaskType.PREDICTION}
    ]
    model = create_hooknet(HOOKNET_CONFIG)
    user_config_dict = insert_paths_into_config(
        SEGMENTATION_CONFIG, image_path, mask_path
    )

    iterator = create_batch_iterator(
        mode="validation",
        user_config=user_config_dict["wholeslidedata"],
        presets=(
            "files",
            "slidingwindow",
        ),
        cpus=1,
        number_of_batches=-1,
        buffer_dtype=np.uint8,
    )

    print("Run inference")
    _execute_inference_single(
        iterator=iterator,
        model=model,
        image_path=image_path,
        files=files,
        output_folder=output_folder,
        tmp_folder=tmp_folder,
    )
    iterator.stop()


if __name__ == "__main__":
    run_segmentation()
