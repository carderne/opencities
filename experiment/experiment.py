from os.path import dirname, join
from random import sample

import rastervision as rv
from pystac import STAC_IO, Catalog
from rastervision.backend.api import PYTORCH_SEMANTIC_SEGMENTATION
from rastervision.utils.files import file_exists

from .constants import TRAIN_IDS, VALID_IDS
from .utils import str_to_bool, my_read_method, my_write_method, read_list

STAC_IO.read_text_method = my_read_method
STAC_IO.write_text_method = my_write_method


class Experiment(rv.ExperimentSet):
    def exp_experiment(
        self,
        experiment_id,
        root_uri,
        train_stac_uri,
        test_stac_uri,
        train_img_dir,
        test_img_dir,
        test_exclude=None,
        test=False,
    ):

        test = str_to_bool(test)
        train_ids = TRAIN_IDS
        valid_ids = VALID_IDS

        train_stac = Catalog.from_file(train_stac_uri)
        test_stac = Catalog.from_file(test_stac_uri)
        all_test_items = test_stac.get_all_items()
        if test_exclude and len(str(test_exclude)) > 2:
            test_exclude = read_list(test_exclude)
        else:
            test_exclude = []

        # Configure chip creation
        chip_opts = {
            "window_method": "sliding",  # use sliding window method of to create chips
            "stride": 512,  # slide over 300px to generate each new chip
        }

        # Training config
        config = {
            "batch_size": 32,  # run out of memory larger than 8
            "num_epochs": 6,  # (originally 6)
            "debug": True,  # produce example chips to help with debugging
            "lr": 1e-4,  # set max lr (originally 1e-4)
            "one_cycle": True,  # use cyclic learning rate scheduler
            "model_arch": "resnet50",  # model architecture
            "loss_fn": "JaccardLoss",
            "augmentors": ["RandomSizedCrop"],
        }

        # Use smaller subset and quicker options for test runs
        if test:
            train_ids = {"acc": {"d41d81": "all"}}
            valid_ids = {"acc": {"a42435": "all"}}
            config["batch_size"] = 2
            config["num_epochs"] = 4
            chip_opts = {"window_method": "random_sample", "chips_per_scene": 4}
            all_test_items = [next(all_test_items) for _ in range(30)]
            experiment_id += "-TEST"

        classes = {"No Building": (2, "#ff00ff"), "Building": (1, "#e6194b")}

        # Create train, validation and test scenes
        print("Creating train scenes")
        train_scenes = [
            make_train_scenes(
                train_stac.get_child(area).get_item(item),
                train_stac_uri,
                train_img_dir,
                which,
            )
            for area, sub in train_ids.items()
            for item, which in sub.items()
        ]
        train_scenes = [item for sublist in train_scenes for item in sublist]

        print("Creating validation scenes")
        valid_scenes = [
            make_train_scenes(
                train_stac.get_child(area).get_item(item),
                train_stac_uri,
                train_img_dir,
                which,
            )
            for area, sub in valid_ids.items()
            for item, which in sub.items()
        ]
        valid_scenes = [item for sublist in valid_scenes for item in sublist]

        # Take random sample of validation scenes
        if len(valid_scenes) > 30:
            valid_scenes = sample(valid_scenes, 30)

        print("Creating test scenes")
        test_scenes = [
            make_test_scene(item, test_img_dir)
            for item in all_test_items
            if item.id not in test_exclude
        ]
        print("Num test scenes: ", len(test_scenes))

        if test:
            train_scenes = train_scenes[:3]
            valid_scenes = valid_scenes[:3]

        # Configure the semantic segmentation task
        task = (
            rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION)
            .with_classes(classes)
            .with_chip_size(512)
            .with_chip_options(**chip_opts)
            .with_predict_chip_size(512)
            .build()
        )

        # Create pytorch backend with configured task
        backend = (
            rv.BackendConfig.builder(PYTORCH_SEMANTIC_SEGMENTATION)
            .with_task(task)
            .with_train_options(**config)
            .with_pretrained_uri("s3://carderne-rv/train/attempt3/model")
            .build()
        )
        # Create DataSet with train, validation and test scenes
        print("Building dataset config")
        dataset = (
            rv.DatasetConfig.builder()
            .with_train_scenes(train_scenes)
            .with_validation_scenes(valid_scenes)
            .with_test_scenes(test_scenes)
            .build()
        )

        # Postprocess to convert background values from 2 to 0
        postprocess_config = {
            "POSTPROCESS": {
                "key": "postprocess",
                "config": {
                    "uris": [
                        join(root_uri, "predict", experiment_id, f"{scene.id}.tif")
                        for scene in test_scenes
                    ],
                    "root_uri": root_uri,
                    "experiment_id": experiment_id,
                },
            }
        }

        # Finally build an experiment from all of these constituent parts.
        experiment = (
            rv.ExperimentConfig.builder()
            .with_id(experiment_id)
            .with_task(task)
            .with_backend(backend)
            .with_dataset(dataset)
            .with_root_uri(root_uri)
            .with_chip_key("chips-512")
            .with_custom_config(postprocess_config)
            .build()
        )

        return experiment


def make_train_window(raster_uri, label_uri, idd, i):
    raster_source = (
        rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE)
        .with_uri(raster_uri)
        .with_channel_order([0, 1, 2])
        .build()
    )
    label_raster_source = (
        rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE)
        .with_vector_source(label_uri)
        .with_rasterizer_options(2)
        .build()
    )
    label_source = (
        rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION)
        .with_raster_source(label_raster_source)
        .build()
    )
    scene = (
        rv.SceneConfig.builder()
        .with_id(f"{idd}_{i}")
        .with_raster_source(raster_source)
        .with_label_source(label_source)
        .build()
    )
    return scene


def make_train_scenes(item, train_stac_uri, train_img_dir, which="all"):
    area = item.get_parent().id
    label_uri = join(
        dirname(train_stac_uri), area, f"{item.id}-labels", f"{item.id}.geojson"
    )

    scenes = []
    if which == "all":
        i = 0
        images_remaining = True
        while images_remaining:
            raster_uri = join(train_img_dir, area, item.id, f"{item.id}_{i}.tif")
            if file_exists(raster_uri):
                scenes.append(make_train_window(raster_uri, label_uri, item.id, i))
            else:
                images_remaining = False
            i += 1
    else:
        for i in which:
            raster_uri = join(train_img_dir, area, item.id, f"{item.id}_{i}.tif")
            scenes.append(make_train_window(raster_uri, label_uri, item.id, i))

    return scenes


def make_test_scene(item, test_img_dir):
    raster_uri = join(test_img_dir, item.id, f"{item.id}.tif")
    raster_source = (
        rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE)
        .with_uri(raster_uri)
        .with_channel_order([0, 1, 2])
        .build()
    )
    scene = (
        rv.SceneConfig.builder()
        .with_id(item.id)
        .with_raster_source(raster_source)
        .build()
    )

    return scene
