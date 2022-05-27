

def create_tumor_bulk_mask():
    wsi = WholeSlideImage(f'/home/user/tempoutput/segoutput/{slide_file}', backend='asap')
    wsa = WholeSlideAnnotation(f'/home/user/tempoutput/bulkoutput/{slide_file[:-4]}.xml')
    if wsa.annotations:
        write_mask(wsi, wsa, spacing=0.5, suffix='_bulk.tif')
    else:
        print('Could not write mask')
        self.predict_in_mask()
        self.detection(slide_file, predictor)
        write_json(0.0, f'{self.output_folder}/{slide_file[:-4]}.json')
        continue


def create_tumor_stroma_mask():
    """Create a mask out of the stroma segmentations within the tumor bulk"""
    folders_to_yml(segpath, bulkpath, self.bulk_config)
    bulk_iterator = create_batch_iterator(
        mode="training",
        user_config=f"{self.bulk_config}/slidingwindowconfig.yml",
        presets=("slidingwindow",),
        cpus=1,
        number_of_batches=-1,
        return_info=True,
    )

    """Write stromal tissue within the tumor bulk to a new tissue mask"""
    spacing = 0.5
    tile_size = 512  # always **2
    current_image = None


    for x_batch, y_batch, info in bulk_iterator:

        filekey = info["sample_references"][0]["reference"].file_key
        if current_image != filekey:
            if current_image:
                bulk_wsm_writer.save()

            current_image = filekey
            slide = image_folder + filekey + ".tif"
            with WholeSlideImage(slide) as wsi:
                shape = wsi.shapes[wsi.get_level_from_spacing(spacing)]
                spacing = wsi.get_real_spacing(spacing)

            bulk_wsm_writer = WholeSlideMaskWriter()
            bulk_wsm_writer.write(
                path=f"tempoutput/detoutput/{filekey}.tif",
                spacing=spacing,
                dimensions=shape,
                tile_shape=(tile_size, tile_size),
            )

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


def create_tumor_stroma_mask():

    # create tumor bulk
    concave_hull(
        input_file=glob.glob("tempoutput/segoutput/*.tif")[0],
        output_dir="tempoutput/bulkoutput/",
        input_level=6,
        output_level=0,
        level_offset=0,
        alpha=0.07,
        min_size=1.5,
        bulk_class=1,
    )

    create_tumor_bulk_mask()
    create_tumor_stroma_mask()
