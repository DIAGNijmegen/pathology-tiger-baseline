from .constants import BULK_CONFIG, BULK_XML_PATH
from .concavehull import concave_hull
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from wholeslidedata.accessories.asap.imagewriter import write_mask


def create_tumor_bulk_mask(image_path, annotation_path):
    wsi = WholeSlideImage(image_path, backend='asap')
    wsa = WholeSlideAnnotation(annotation_path)

    if len(wsa.annotations) == 0:
        raise RuntimeError('Could not generate valid tumor bulk')

    print('Creating tumor mask')
    write_mask(wsi, wsa, spacing=0.5, suffix='.tif')


def _create_tumor_stroma_mask(segmentation_path, bulk_path):
    """Create a mask out of the stroma segmentations within the tumor bulk"""
   
    user_config_dict = insert_paths_into_config(
        BULK_CONFIG, segmentation_path, bulk_path

    )

    bulk_iterator = create_batch_iterator(
        mode="training",
        user_config=user_config_dict["wholeslidedata"],
        presets=("files", "slidingwindow",),
        cpus=1,
        number_of_batches=-1,
        return_info=True,
    )

    """Write stromal tissue within the tumor bulk to a new tissue mask"""
    spacing = 0.5
    tile_size = 1024  # always **2

    with WholeSlideImage(segmentation_path, backend='asap') as wsi:
        shape = wsi.shapes[wsi.get_level_from_spacing(spacing)]
        spacing = wsi.get_real_spacing(spacing)

    bulk_wsm_writer = WholeSlideMaskWriter()
    bulk_wsm_writer.write('tumor_stroma_mask.tif', spacing=spacing, dimensions=shape, tile_shape=(tile_size, tile_size))


    for x_batch, y_batch, info in bulk_iterator:

        points = [x["point"] for x in info["sample_references"]]

        if len(points) != len(x_batch):
            points = points[: len(x_batch)]

        for i, point in enumerate(points):
            c, r = point.x, point.y
            x_batch[i][x_batch[i] == 1] = 0
            x_batch[i][x_batch[i] == 3] = 0
            x_batch[i][x_batch[i] == 5] = 0
            x_batch[i][x_batch[i] == 4] = 0
            x_batch[i][x_batch[i] == 7] = 0
            new_mask = x_batch[i].reshape(tile_size, tile_size) * y_batch[i]
            new_mask[new_mask > 0] = 1
            bulk_wsm_writer.write_tile(
                tile=new_mask, coordinates=(int(c), int(r)), mask=y_batch[i]
            )

    bulk_wsm_writer.save()
    bulk_iterator.stop()


def create_tumor_stroma_mask(segmentation_path):

    # # create tumor bulk
    # concave_hull(
    #     input_file=segmentation_path,
    #     output_dir="/home/user/tmp/",
    #     input_level=6,
    #     output_level=0,
    #     level_offset=0,
    #     alpha=0.07,
    #     min_size=1.5,
    #     bulk_class=1,
    # )

    create_tumor_bulk_mask(segmentation_path, BULK_XML_PATH)
    create_tumor_stroma_mask()
