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

SOURCE_CONFIG = Path('configs/inputs.yml')

HOOKNET_WEIGHTS = _ALGORITHM_FOLDER / 'configs' / 'segmentation'
HOOKNET_CONFIG = Path("")
SEGMENTATION_CONFIG = Path("")

DETECTION_WEIGHTS = Path("")
DETECTION_CONFIG = Path("")

BULK_CONFIG = Path("")
