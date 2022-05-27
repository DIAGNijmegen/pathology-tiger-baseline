import traceback
from pathlib import Path

from wholeslidedata.source.configuration.config import get_paths

from .constants import (ASAP_DETECTION_OUTPUT, DETECTION_OUTPUT_PATH, OUTPUT_FOLDER,
                        SEGMENTATION_OUTPUT_PATH, SOURCE_CONFIG, TILS_OUTPUT_PATH,
                        TUMOR_STROMA_MASK_PATH)
from .detection import run_detection
from .segmentation import run_segmentation
from .tilscore import create_til_score
from .tumorstroma import create_tumor_stroma_mask
from .utils import is_l1, write_json
import subprocess

def create_lock_file(lock_file_path):
    print(f"Creating lock file: {lock_file_path}")
    Path(lock_file_path).touch()


def release_lock_file(lock_file_path):
    print(f"Releasing lock file {lock_file_path}")
    Path(lock_file_path).unlink(missing_ok=True)


def process_l1(image_path, mask_path):
    run_detection(image_path, mask_path)


def process_l2(image_path):
    create_tumor_stroma_mask(SEGMENTATION_OUTPUT_PATH)
    run_detection(image_path, TUMOR_STROMA_MASK_PATH)
    create_til_score(image_path, ASAP_DETECTION_OUTPUT)

def cleanup():

    det_result = dict(
        type="Multiple points", points=[], version={"major": 1, "minor": 0}
    )
    write_json(det_result, DETECTION_OUTPUT_PATH)
    write_json(0.0, TILS_OUTPUT_PATH)

def main():
    for image_path, mask_path in get_paths(SOURCE_CONFIG, preset="folders"):
        print(f"PROCESSING: {image_path}, with {mask_path}....")

        lock_file_path = OUTPUT_FOLDER / (image_path.stem + ".lock")
        if lock_file_path.exists():
            print("Lock file exists, skipping inference.")
            continue
        try:
            create_lock_file(lock_file_path=lock_file_path)
            

            print('running segmentation')
            cmd = ['python3', '-m', 'baseline.segmentation',
                   f'--image_path={image_path}',
                   f'--mask_path={mask_path}']
            
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

            p.wait()
            if p.stderr is not None:
                print('/n'.join([line.decode("utf-8") for line in p.stderr.readlines()]))

            print('segmentation done')

            if is_l1(mask_path):
                process_l1(image_path, mask_path)
            else:
                process_l2(image_path)

        except Exception as e:
            print("Exception")
            print(e)
            cleanup()
            print(traceback.format_exc())
        finally:
            release_lock_file(lock_file_path=lock_file_path)
        print("--------------")


if __name__ == "__main__":
    main()
