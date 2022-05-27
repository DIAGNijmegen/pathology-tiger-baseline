from pathlib import Path

_ALGORITHM_FOLDER = Path(__file__).parent

OUTPUT_FOLDER = Path("/output/")

SEGMENTATION_OUTPUT_PATH = Path("/output/images/breast-cancer-segmentation-for-tils/segmentation.tif")
DETECTION_OUTPUT_PATH = Path("/output/detected-lymphocytes.json")
TILS_OUTPUT_PATH = Path("/output/til-score.json")

### Temporary paths
TMP_FOLDER = Path("/home/user/tmp")
TMP_SEGMENTATION_OUTPUT_PATH = TMP_FOLDER / SEGMENTATION_OUTPUT_PATH.name
TMP_DETECTION_OUTPUT_PATH = TMP_FOLDER / DETECTION_OUTPUT_PATH.name
TMP_TILS_SCORE_PATH = TMP_FOLDER / TILS_OUTPUT_PATH.name

ASAP_DETECTION_OUTPUT = TMP_FOLDER / "detections_asap.xml"
BULK_XML_PATH = TMP_FOLDER / "tumorbulk.xml"
BULK_MASK_PATH = TMP_FOLDER / "tumorbulk.tif"
TUMOR_STROMA_MASK_PATH = TMP_FOLDER / "tumor_stroma_mask.tif"

# CONFIGS
SOURCE_CONFIG = _ALGORITHM_FOLDER / "configs" / "inputs.yml"
HOOKNET_CONFIG = _ALGORITHM_FOLDER / "configs" / "segmentation" / "hooknet_params.yml"
SEGMENTATION_CONFIG = _ALGORITHM_FOLDER / "configs" / "segmentation" / "hooknet_segmentation.yml"
DETECTION_CONFIG = _ALGORITHM_FOLDER / "configs" / "detection" / "detectron2_detection.yml"
BULK_CONFIG = _ALGORITHM_FOLDER / "configs" / "bulk" / "bulk.yml"