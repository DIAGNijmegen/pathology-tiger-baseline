from pathlib import Path

_ALGORITHM_FOLDER = Path(__file__).parent

OUTPUT_FOLDER = Path("/output/")
SEGMENTATION_OUTPUT_FOLDER = OUTPUT_FOLDER / 'images' / 'breast-cancer-segmentation-for-tils'

### Temporary paths
TMP_FOLDER = Path("/home/user/tmp")

ASAP_DETECTION_OUTPUT = TMP_FOLDER / "detections_asap.xml"
BULK_XML_PATH = TMP_FOLDER / "tumorbulk.xml"
BULK_MASK_PATH = TMP_FOLDER / "tumorbulk.tif"
TUMOR_STROMA_MASK_PATH = TMP_FOLDER / "tumor_stroma_mask.tif"

# CONFIGS
GRAND_CHALLENGE_SOURCE_CONFIG = _ALGORITHM_FOLDER / "configs" / "grandchallengeinputs.yml"
HOOKNET_CONFIG = _ALGORITHM_FOLDER / "configs" / "segmentation" / "hooknet_params.yml"
SEGMENTATION_CONFIG = _ALGORITHM_FOLDER / "configs" / "segmentation" / "hooknet_segmentation.yml"
DETECTION_CONFIG = _ALGORITHM_FOLDER / "configs" / "detection" / "detectron2_detection.yml"
BULK_CONFIG = _ALGORITHM_FOLDER / "configs" / "bulk" / "bulk.yml"