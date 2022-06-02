import json

import numpy as np
from tqdm import tqdm
from wholeslidedata.accessories.asap.annotationwriter import write_point_set
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_batch_iterator
from wholeslidedata.source.configuration.config import insert_paths_into_config

from baseline.constants import ASAP_DETECTION_OUTPUT, DETECTION_CONFIG
from baseline.wsdetectron2 import Detectron2DetectionPredictor

from .nms import to_wsd
from .utils import px_to_mm


def inference(iterator, predictor, image_path, output_path):

    print("predicting...")
    output_dict = {
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    annotations = []

    with WholeSlideImage(image_path) as wsi:
        spacing = wsi.get_real_spacing(0.5)
    print(iterator.dataset.annotation_counts)

    if predictor is None:
        predictor = Detectron2DetectionPredictor(
            output_dir="/home/user/tmp/",
            threshold=0.1,
            nms_threshold=0.3,
        )

    for x_batch, y_batch, info in tqdm(iterator):
        predictions = predictor.predict_on_batch(x_batch)
        for idx, prediction in enumerate(predictions):
            point = info["sample_references"][idx]["point"]
            c, r = point.x, point.y

            for detections in prediction:
                x, y, label, confidence = detections.values()

                if x == 128 or y == 128:
                    continue

                if y_batch[idx][y][x] == 0:
                    continue
                
                x += c
                y += r
                prediction_record = {
                    "point": [
                        px_to_mm(x, spacing),
                        px_to_mm(y, spacing),
                        0.5009999871253967,
                    ],
                    "probability": confidence,
                }
                output_dict["points"].append(prediction_record)
                annotations.append((x, y))

    print(f"Predicted {len(annotations)} points")
    print("saving predictions...")

    annotations = to_wsd(annotations)

    write_point_set(
        annotations,
        ASAP_DETECTION_OUTPUT,
        label_name="lymphocytes",
        label_color="blue",
    )

    with open(output_path, "w") as outfile:
        json.dump(output_dict, outfile, indent=4)

    print("finished!")


def run_detection(model, image_path, mask_path, output_path):

    print(image_path, mask_path)

    user_config_dict = insert_paths_into_config(DETECTION_CONFIG, image_path, mask_path)

    iterator = create_batch_iterator(
        mode="validation",
        user_config=user_config_dict["wholeslidedata"],
        presets=(
            "files",
            "slidingwindow",
        ),
        cpus=1,
        number_of_batches=-1,
        return_info=True,
        buffer_dtype=np.uint8,
    )

    inference(
        iterator=iterator,
        predictor=model,
        image_path=image_path,
        output_path=output_path,
    )
    iterator.stop()
