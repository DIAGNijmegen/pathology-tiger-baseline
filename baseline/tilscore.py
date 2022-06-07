from .utils import dist_to_px, get_mask_area, write_json
from .nms import slide_nms, to_wsd
from .constants import TUMOR_STROMA_MASK_PATH

def create_til_score(image_path, xml_path, output_path):
    """slide_level_nms"""
    points = slide_nms(image_path, xml_path, 256)
    wsd_points = to_wsd(points)
    print(len(wsd_points))
    """Compute TIL score and write to output"""
    til_area = dist_to_px(8, 0.5) ** 2
    tils_area = len(wsd_points) * til_area
    stroma_area = get_mask_area(TUMOR_STROMA_MASK_PATH)
    tilscore = (100 / int(stroma_area)) * int(tils_area)
    print(f"tilscore = {tilscore}")
    write_json(tilscore, output_path)