import os
from os.path import basename, join
from subprocess import call

import numpy as np
import rasterio
import rastervision as rv
from rasterio.windows import Window
from rastervision.utils.files import upload_or_copy

PREPROCESS = "PREPROCESS"


def split_image(image_uri, split_dir):
    with rasterio.open(image_uri) as src:
        width = src.width
        height = src.height

    win_size = 8704
    wins = []
    print("Splitting image", image_uri)
    for c in list(range(0, width, win_size)):
        if c >= width:
            continue
        if (c + win_size) > width:
            win_width = width - c - 1
        else:
            win_width = win_size
        for r in list(range(0, height, win_size)):
            if r >= height:
                continue
            if (r + win_size) > height:
                win_height = height - r - 1
            else:
                win_height = win_size
            wins.append(Window(c, r, win_width, win_height))

    i = 0
    for win in wins:
        print("Doing window", i, " params:   ", win)
        with rasterio.open(image_uri) as src:
            img = src.read(window=win)
            win_transform = src.window_transform(win)
            kwargs = src.meta.copy()

        if np.max(img[-1]) == 255:
            area = image_uri.split("/")[-3]
            image_id = image_uri.split("/")[-2]
            output_uri = join(split_dir, area, image_id, f"{image_id}_{i}.tif")
            kwargs.update(
                {"height": win.height, "width": win.width, "transform": win_transform}
            )

            tmp_uri = join("/tmp/", basename(output_uri))
            with rasterio.open(tmp_uri, "w", **kwargs) as dst:
                dst.write(img)
            tmp_cmpr_file = tmp_uri.replace(".tif", "_jpg.tif")
            gdal_command = (
                f"gdal_translate {tmp_uri} {tmp_cmpr_file} "
                f"-co COMPRESS=JPEG -co JPEG_QUALITY=100 -co TILED=YES "
                f"-co COPY_SRC_OVERVIEWS=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 "
                f"--config COMPRESS_OVERVIEW JPEG"
            )
            call(gdal_command, shell=True)
            upload_or_copy(tmp_cmpr_file, output_uri)
            for t in (tmp_uri, tmp_cmpr_file):
                os.remove(t)
            i += 1


class PreProcessCommand(rv.AuxCommand):
    command_type = PREPROCESS
    options = rv.AuxCommandOptions(
        split_on="items",
        inputs=lambda conf: PreProcessCommand.gather_inputs(conf),
        outputs=lambda conf: PreProcessCommand.gather_outputs(conf),
        required_fields=["items", "split_dir"],
    )

    def run(self):
        for image_uri in self.command_config["items"]:
            split_image(image_uri, self.command_config["split_dir"])

    @staticmethod
    def gather_inputs(conf):
        return conf["items"]

    @staticmethod
    def gather_outputs(conf):
        return [".phony"]


def register_plugin(plugin_registry):
    plugin_registry.register_aux_command(PREPROCESS, PreProcessCommand)
