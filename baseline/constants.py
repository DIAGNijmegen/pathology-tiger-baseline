from pathlib import Path

_ALGORITHM_FOLDER = Path(__file__).parent

OUTPUT_FOLDER = Path('/output/')

SEGMENTATION_OUTPUT_PATH = Path(
    "/output/images/breast-cancer-segmentation-for-tils/segmentation.tif"
)
DETECTION_OUTPUT_PATH = Path("/output/detected-lymphocytes.json")
TILS_OUTPUT_PATH = Path("/output/til-score.json")

### Temporary paths
TMP_FOLDER = Path("/home/user/tmp")
TMP_SEGMENTATION_OUTPUT_PATH = TMP_FOLDER / SEGMENTATION_OUTPUT_PATH.name
TMP_DETECTION_OUTPUT_PATH = TMP_FOLDER / DETECTION_OUTPUT_PATH.name
TMP_TILS_SCORE_PATH = TMP_FOLDER / TILS_OUTPUT_PATH.name

SOURCE_CONFIG = _ALGORITHM_FOLDER / Path('configs/inputs.yml')

HOOKNET_CONFIG = _ALGORITHM_FOLDER / "configs" / "segmentation" / "hooknet_params.yml"
SEGMENTATION_CONFIG = _ALGORITHM_FOLDER / "configs" / "segmentation" / "hooknet_segmentation.yml"

DETECTION_CONFIG = _ALGORITHM_FOLDER / "configs" / "detection" / "detectron2_detection.yml"

BULK_XML_PATH = Path('/home/user/tmp/tumorbulk.xml')
BULK_CONFIG =_ALGORITHM_FOLDER / 'configs' / 'bulk' / 'bulk.yml'