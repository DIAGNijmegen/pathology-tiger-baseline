import numpy as np
from hooknet.configuration.config import create_hooknet
from hooknet.inference.apply import _execute_inference_single
from hooknet.inference.writing import MaskType
from wholeslidedata.iterators import create_batch_iterator
from wholeslidedata.source.configuration.config import insert_paths_into_config

from .constants import HOOKNET_CONFIG, SEGMENTATION_CONFIG


def run_segmentation(model, image_path, mask_path, output_folder, tmp_folder, name):
    files = [{"name": name, "type": MaskType.PREDICTION}]
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
