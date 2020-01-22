from functools import reduce
from os.path import dirname, join
from random import sample

import rastervision as rv
from pystac import STAC_IO, Catalog
from rastervision.backend.api import PYTORCH_SEMANTIC_SEGMENTATION
from rastervision.utils.files import file_exists

from .constants import TRAIN_IDS, VALID_IDS
from .utils import str_to_bool, my_read_method, my_write_method

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
        test=False,
    ):

        test = str_to_bool(test)
        train_ids = TRAIN_IDS
        valid_ids = VALID_IDS

        train_stac = Catalog.from_file(train_stac_uri)
        test_stac = Catalog.from_file(test_stac_uri)
        all_test_items = test_stac.get_all_items()

        # Configure chip creation
        chip_opts = {
            "window_method": "sliding",  # use sliding window method of to create chips
            "stride": 300,  # slide over 300px to generate each new chip
        }

        # Training configuration: try to improve performance by tuning these
        config = {
            "batch_size": 8,  # 8 chips per batch
            "num_epochs": 6,  # complete 6 epochs
            "debug": True,  # produce example chips to help with debugging
            "lr": 1e-4,  # set learning
            "one_cycle": True,  # use cyclic learning rate scheduler
            "model_arch": "resnet18",  # model architecture
            "loss_fn": "JaccardLoss",
        }

        if test:
            train_ids = sample(train_ids, 2)
            valid_ids = sample(valid_ids, 2)

            config["batch_size"] = 2
            config["num_epochs"] = 1

            # In this 'test' scenario, we will generate a small number of chips
            chip_opts = {"window_method": "random_sample", "chips_per_scene": 10}

            # Test only 5 items
            all_test_items = [next(all_test_items) for _ in range(5)]

            # Modify the experiment ID
            experiment_id += "-TEST"

        def make_train_scenes(item):
            area = item.get_parent().id
            label_uri = join(
                dirname(train_stac_uri), area, f"{item.id}-labels", f"{item.id}.geojson"
            )

            i = 0
            images_remaining = True
            scenes = []
            while images_remaining:
                raster_uri = join(train_img_dir, area, item.id, f"{item.id}_{i}.tif")
                if file_exists(raster_uri):
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
                        .with_task(task)
                        .with_id(f"{item.id}_{i}")
                        .with_raster_source(raster_source)
                        .with_label_source(label_source)
                        .build()
                    )
                    scenes.append(scene)
                else:
                    images_remaining = False
                i += 1

            return scenes

        def make_test_scene(item):
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

        classes = {"No Building": (2, "#ff00ff"), "Building": (1, "#e6194b")}
        # Configure the semantic segmentation task
        task = (
            rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION)
            .with_classes(classes)
            .with_chip_options(**chip_opts)
            .build()
        )

        # Create pytorch backend with configured task
        backend = (
            rv.BackendConfig.builder(PYTORCH_SEMANTIC_SEGMENTATION)
            .with_task(task)
            .with_train_options(**config)
            .build()
        )

        # Create train, validation and test scenes
        print("Creating train scenes")
        train_scenes = reduce(
            lambda a, b: a + b,
            [
                make_train_scenes(train_stac.get_child(c).get_item(i))
                for c, i in train_ids
            ],
        )

        print("Creating validation scenes")
        valid_scenes = reduce(
            lambda a, b: a + b,
            [
                make_train_scenes(train_stac.get_child(c).get_item(i))
                for c, i in valid_ids
            ],
        )
        # Take random sample of validation scenes
        if len(valid_scenes) > 30:
            valid_scenes = sample(valid_scenes, 30)

        print("Creating test scenes")
        test_scenes = [make_test_scene(item) for item in all_test_items]

        if test:
            train_scenes = sample(train_scenes, 3)
            valid_scenes = sample(valid_scenes, 3)

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
            .with_custom_config(postprocess_config)
            .build()
        )

        return experiment
