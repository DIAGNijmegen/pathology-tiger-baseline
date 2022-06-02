import traceback
from pathlib import Path

from wholeslidedata.source.configuration.config import get_paths
import os, shutil

from baseline.wsdetectron2 import Detectron2DetectionPredictor
from .tilscore import create_til_score
from .tumorstroma import create_tumor_stroma_mask
from .utils import is_l1, write_json
from .segmentation import run_segmentation
from .detection import run_detection
import click
from hooknet.configuration.config import create_hooknet
from .constants import (
    ASAP_DETECTION_OUTPUT,
    BULK_MASK_PATH,
    BULK_XML_PATH,
    GRAND_CHALLENGE_SOURCE_CONFIG,
    HOOKNET_CONFIG,
    OUTPUT_FOLDER,
    SEGMENTATION_OUTPUT_FOLDER,
    TMP_FOLDER,
    TUMOR_STROMA_MASK_PATH,
)
import tensorflow as tf
import tensorflow.keras.backend as K
import torch
import gc


def create_lock_file(lock_file_path):
    print(f"Creating lock file: {lock_file_path}")
    Path(lock_file_path).touch()


def release_lock_file(lock_file_path):
    print(f"Releasing lock file {lock_file_path}")
    Path(lock_file_path).unlink(missing_ok=True)


def write_empty_files(detection_output_path, tils_output_path):
    det_result = dict(
        type="Multiple points", points=[], version={"major": 1, "minor": 0}
    )
    write_json(det_result, detection_output_path)
    write_json(0.0, tils_output_path)


def get_source_config(image_folder, mask_folder):
    return {
        "source": {
            "default": {
                "image_sources": {"folder": str(image_folder)},
                "annotation_sources": {"folder": str(mask_folder)},
            }
        }
    }


def delete_tmp_files():
    for filename in os.listdir(str(TMP_FOLDER)):
        file_path = os.path.join(str(TMP_FOLDER), filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def set_tf_gpu_config():
    """Hard-coded GPU limit to balance between tensorflow and Pytorch"""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6024)]
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def tf_be_silent():
    """Surpress exessive TF warnings"""
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.get_logger().setLevel("ERROR")
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception as ex:
        print("failed to silence tf warnings:", str(ex))


@click.command()
@click.option("--source_config", type=Path, required=False)
@click.option("--image_folder", type=Path, required=False)
@click.option("--mask_folder", type=Path, required=False)
@click.option("--grandchallenge", type=bool, default=True, required=False)
def main(
    source_config: Path = None,
    image_folder: Path = None,
    mask_folder: Path = None,
    grandchallenge: bool = True,
):

    set_tf_gpu_config()
    tf_be_silent()
    torch.cuda.set_per_process_memory_fraction(0.55, 0)

    if grandchallenge:
        segmentation_model = None
        detection_model = None
    else:
        segmentation_model = create_hooknet(HOOKNET_CONFIG)
        detection_model = Detectron2DetectionPredictor(
            output_dir="/home/user/tmp/",
            threshold=0.1,
            nms_threshold=0.3,
        )

    if source_config is None and image_folder is None and mask_folder is None:
        source_config = GRAND_CHALLENGE_SOURCE_CONFIG
    elif image_folder is not None and mask_folder is not None:
        source_config = get_source_config(
            image_folder=image_folder, mask_folder=mask_folder
        )

    for image_path, mask_path in get_paths(source_config, preset="folders"):
        print(f"PROCESSING: {image_path}, with {mask_path}....")

        segmentation_file_name = image_path.stem + "_tiger_baseline.tif"
        segmentation_path = SEGMENTATION_OUTPUT_FOLDER / segmentation_file_name
        if grandchallenge:
            detection_output_path = OUTPUT_FOLDER / "detected-lymphocytes.json"
            tils_output_path = OUTPUT_FOLDER / "til-score.json"
        else:
            detection_output_path = (
                OUTPUT_FOLDER / f"{image_path.stem}_detected-lymphocytes.json"
            )
            tils_output_path = OUTPUT_FOLDER / f"{image_path.stem}_til-score.json"

        lock_file_path = OUTPUT_FOLDER / (image_path.stem + ".lock")
        if lock_file_path.exists():
            print("Lock file exists, skipping inference.")
            continue
        try:
            create_lock_file(lock_file_path=lock_file_path)

            SEGMENTATION_OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
            run_segmentation(
                model=segmentation_model,
                image_path=image_path,
                mask_path=mask_path,
                output_folder=SEGMENTATION_OUTPUT_FOLDER,
                tmp_folder=TMP_FOLDER,
                name=segmentation_file_name,
            )
            if grandchallenge:
                K.clear_session()
                del segmentation_model 
            gc.collect()    

            if is_l1(mask_path):
                print("L1")
                run_detection(
                    model=detection_model,
                    image_path=image_path,
                    mask_path=mask_path,
                    output_path=detection_output_path,
                )
                gc.collect()
                write_json(0.0, tils_output_path)
            else:
                print("L2")
                create_tumor_stroma_mask(
                    segmentation_path=segmentation_path,
                    bulk_xml_path=BULK_XML_PATH,
                    bulk_mask_path=BULK_MASK_PATH,
                )
                gc.collect()
                run_detection(
                    model=detection_model,
                    image_path=image_path,
                    mask_path=TUMOR_STROMA_MASK_PATH,
                    output_path=detection_output_path,
                )
                gc.collect()
                create_til_score(
                    image_path=image_path,
                    xml_path=ASAP_DETECTION_OUTPUT,
                    output_path=tils_output_path,
                )

        except Exception as e:
            print("Exception")
            print(e)
            write_empty_files(
                detection_output_path=detection_output_path,
                tils_output_path=tils_output_path,
            )
            print(traceback.format_exc())
        finally:
            delete_tmp_files()
            release_lock_file(lock_file_path=lock_file_path)
        print("--------------")


if __name__ == "__main__":
    main()
