import traceback
from pathlib import Path

from wholeslidedata.source.configuration.config import get_paths

from .constants import OUTPUT_FOLDER, SOURCE_CONFIG, SEGMENTATION_OUTPUT_PATH
from .detection import run_detection
from .segmentation import run_segmentation
from .tilscore import create_til_score
from .tumorstroma import create_tumor_stroma_mask


def create_lock_file(lock_file_path):
    print(f"Creating lock file: {lock_file_path}")
    Path(lock_file_path).touch()


def release_lock_file(lock_file_path):
    print(f"Releasing lock file {lock_file_path}")
    Path(lock_file_path).unlink(missing_ok=True)


def process_l1(image_path, mask_path):
    run_segmentation(image_path, mask_path)
    run_detection(image_path, mask_path)


def process_l2(image_path, mask_path):
    # run_segmentation(image_path, mask_path)
    create_tumor_stroma_mask(SEGMENTATION_OUTPUT_PATH)
    
    # run_detection(image_path, )
    # create_til_score()


def is_l1():
    return False

def cleanup():
    pass
    # create empty json
    # create 0 til score

def main():
    print("Create output folder")
    for image_path, mask_path in get_paths(SOURCE_CONFIG, preset="folders"):
        print(f"PROCESSING: {image_path}, with {mask_path}....")

        lock_file_path = OUTPUT_FOLDER / (image_path.stem + ".lock")
        if lock_file_path.exists():
            print("Lock file exists, skipping inference.")
            continue
        try:
            create_lock_file(lock_file_path=lock_file_path)
            
            if is_l1():
                process_l1(image_path, mask_path)
            else:
                process_l2(image_path, mask_path)

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
