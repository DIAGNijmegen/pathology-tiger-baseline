import traceback
from pathlib import Path

from wholeslidedata.source.configuration.config import get_paths

from .constants import (
    ASAP_DETECTION_OUTPUT,
    BULK_MASK_PATH,
    BULK_XML_PATH,
    DETECTION_OUTPUT_PATH,
    OUTPUT_FOLDER,
    SEGMENTATION_OUTPUT_PATH,
    GRAND_CHALLENGE_SOURCE_CONFIG,
    TILS_OUTPUT_PATH,
    TMP_SEGMENTATION_OUTPUT_PATH,
    TUMOR_STROMA_MASK_PATH,
)
from .detection import run_detection
from .tilscore import create_til_score
from .tumorstroma import create_tumor_stroma_mask
from .utils import is_l1, write_json
import subprocess
import click


def create_lock_file(lock_file_path):
    print(f"Creating lock file: {lock_file_path}")
    Path(lock_file_path).touch()


def release_lock_file(lock_file_path):
    print(f"Releasing lock file {lock_file_path}")
    Path(lock_file_path).unlink(missing_ok=True)


def write_empty_files():
    det_result = dict(
        type="Multiple points", points=[], version={"major": 1, "minor": 0}
    )
    write_json(det_result, DETECTION_OUTPUT_PATH)
    write_json(0.0, TILS_OUTPUT_PATH)


def get_source_config(image_folder, mask_folder):
    return {
        "source": {
            "default": {
                "image_sources": {"folder": str(image_folder)},
                "annotation_sources": {"folder": str(mask_folder)},
            }
        }
    }


@click.command()
@click.option("--source_config", type=Path, required=False)
@click.option("--image_folder", type=Path, required=False)
@click.option("--mask_folder", type=Path, required=False)
def main(
    source_config: Path = None, image_folder: Path = None, mask_folder: Path = None
):

    if source_config is None and image_folder is None and mask_folder is None:
        source_config = GRAND_CHALLENGE_SOURCE_CONFIG
    elif image_folder is not None and mask_folder is not None:
        source_config = get_source_config(
            image_folder=image_folder, mask_folder=mask_folder
        )

    for image_path, mask_path in get_paths(source_config, preset="folders"):
        print(f"PROCESSING: {image_path}, with {mask_path}....")

        lock_file_path = OUTPUT_FOLDER / (image_path.stem + ".lock")
        if lock_file_path.exists():
            print("Lock file exists, skipping inference.")
            continue
        try:
            create_lock_file(lock_file_path=lock_file_path)
            
            SEGMENTATION_OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)

            print("running segmentation")
            cmd = [
                "python3",
                "-u",
                "-m",
                "baseline.segmentation",
                f"--image_path={image_path}",
                f"--mask_path={mask_path}",
                f"--output_folder={SEGMENTATION_OUTPUT_PATH.parent}",
                f"--tmp_folder={TMP_SEGMENTATION_OUTPUT_PATH.parent}",
            ]

            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

            p.wait()
            if p.stderr is not None:
                print(
                    "/n".join([line.decode("utf-8") for line in p.stderr.readlines()])
                )

            print("segmentation done")

            if is_l1(mask_path):
                run_detection(image_path, mask_path)
            else:
                create_tumor_stroma_mask(
                    segmentation_path=SEGMENTATION_OUTPUT_PATH,
                    bulk_xml_path=BULK_XML_PATH,
                    bulk_mask_path=BULK_MASK_PATH,
                )
                run_detection(image_path, TUMOR_STROMA_MASK_PATH)
                create_til_score(image_path, ASAP_DETECTION_OUTPUT)

        except Exception as e:
            print("Exception")
            print(e)
            write_empty_files()
            print(traceback.format_exc())
        finally:
            SEGMENTATION_OUTPUT_PATH.rename(SEGMENTATION_OUTPUT_PATH.parent / (image_path.stem + '.tif'))
            release_lock_file(lock_file_path=lock_file_path)
        print("--------------")


if __name__ == "__main__":
    main()
