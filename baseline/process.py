import traceback
from pathlib import Path

from wholeslidedata.source.configuration.config import get_paths
import os, shutil

from .tilscore import create_til_score
from .tumorstroma import create_tumor_stroma_mask
from .utils import is_l1, write_json
import click
from .constants import (
    ASAP_DETECTION_OUTPUT,
    BULK_MASK_PATH,
    BULK_XML_PATH,
    GRAND_CHALLENGE_SOURCE_CONFIG,
    OUTPUT_FOLDER,
    SEGMENTATION_OUTPUT_FOLDER,
    TMP_FOLDER,
    TUMOR_STROMA_MASK_PATH,
)
import gc
import subprocess

from .utils import timing


def print_std(p: subprocess.Popen):

    if p.stderr is not None:
        for line in p.stderr.readlines():
            print(line)

    if p.stdout is not None:
        for line in p.stdout.readlines():
            print(line)

@timing
def run_segmentation(image_path, mask_path, output_folder, tmp_folder, name):

    print("running segmentation")
    cmd = [
        "python3",
        "-u",
        "-m",
        "baseline.segmentation",
        f"--image_path={image_path}",
        f"--mask_path={mask_path}",
        f"--output_folder={output_folder}",
        f"--tmp_folder={tmp_folder}",
        f"--name={name}",
    ]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p.wait()
    print_std(p)

@timing
def run_detection(image_path, mask_path, output_path):

    print("running detection")
    cmd = [
        "python3",
        "-u",
        "-m",
        "baseline.detection",
        f"--image_path={image_path}",
        f"--mask_path={mask_path}",
        f"--output_path={output_path}",
    ]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p.wait()
    print_std(p)

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


def delete_data_files():
    for filename in os.listdir("/home/user/data/"):
        file_path = os.path.join(str("/home/user/data/"), filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


@click.command()
@click.option("--source_config", type=Path, required=False)
@click.option("--image_folder", type=Path, required=False)
@click.option("--mask_folder", type=Path, required=False)
@click.option("--resection", type=bool, default=True, required=False)
@click.option("--grandchallenge", type=bool, default=True, required=False)
def main(
    source_config: Path = None,
    image_folder: Path = None,
    mask_folder: Path = None,
    resection: bool = True,
    grandchallenge: bool = True,
):


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

        if (
            not grandchallenge
            and segmentation_path.exists()
            and detection_output_path.exists()
            and tils_output_path.exists()
        ):
            print('All files already exists. continue')
            continue

        lock_file_path = OUTPUT_FOLDER / (image_path.stem + ".lock")
        if not grandchallenge and lock_file_path.exists():
            print("Lock file exists, skipping inference.")
            continue
        try:
            if not grandchallenge:
                create_lock_file(lock_file_path=lock_file_path)

            SEGMENTATION_OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
            run_segmentation(
                image_path=image_path,
                mask_path=mask_path,
                output_folder=SEGMENTATION_OUTPUT_FOLDER,
                tmp_folder=TMP_FOLDER,
                name=segmentation_file_name,
            )
            gc.collect()

            if is_l1(mask_path):
                print("L1")
                run_detection(
                    image_path=image_path,
                    mask_path=mask_path,
                    output_path=detection_output_path,
                )
                gc.collect()
                write_json(0.0, tils_output_path)
            else:
                print("L2")
                print('tumor stroma mask')
                create_tumor_stroma_mask(
                    segmentation_path=segmentation_path,
                    bulk_xml_path=BULK_XML_PATH,
                    bulk_mask_path=BULK_MASK_PATH,
                    resection=resection,
                )
                gc.collect()
                print('detection')
                run_detection(
                    image_path=image_path,
                    mask_path=TUMOR_STROMA_MASK_PATH,
                    output_path=detection_output_path,
                )
                gc.collect()
                print('computing til score')
                create_til_score(
                    image_path=image_path,
                    xml_path=ASAP_DETECTION_OUTPUT,
                    output_path=tils_output_path,
                )

        except Exception as e:
            print('Exception')
            print(e)
            print("Writing empty files...")
            # print(e)
            write_empty_files(
                detection_output_path=detection_output_path,
                tils_output_path=tils_output_path,
            )
            # print(traceback.format_exc())
        finally:
            if not grandchallenge:
                delete_tmp_files()
                delete_data_files()
                release_lock_file(lock_file_path=lock_file_path)
        print("--------------")


if __name__ == "__main__":
    main()
