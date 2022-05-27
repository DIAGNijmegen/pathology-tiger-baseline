import traceback
from pathlib import Path

from wholeslidedata.source.configuration.config import get_paths

from .constants import OUTPUT_FOLDER, SOURCE_CONFIG
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


def create_output_folders():
    Path(f"/output/images/breast-cancer-segmentation-for-tils").mkdir(
        parents=True, exist_ok=True
    )
    # TODO


def process_l1():
    run_segmentation()
    run_detection()


def process_l2():
    run_segmentation()
    create_tumor_stroma_mask()
    run_detection()
    create_til_score()


def is_l1():
    return True


def main():
    print("Create output folder")
    create_output_folders()

    for image_path, annotation_path in get_paths(SOURCE_CONFIG, preset="folders"):
        print(f"PROCESSING: {image_path}, with {annotation_path}....")

        lock_file_path = OUTPUT_FOLDER / (image_path.stem + ".lock")
        if lock_file_path.exists():
            print("Lock file exists, skipping inference.")
            continue
        try:
            create_lock_file(lock_file_path=lock_file_path)

            if is_l1():
                process_l1()
            else:
                process_l2()

        except Exception as e:
            print("Exception")
            print(e)
            print(traceback.format_exc())
        finally:
            release_lock_file(lock_file_path=lock_file_path)
        print("--------------")


if __name__ == "__main__":
    main()
