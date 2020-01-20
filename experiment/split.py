from os.path import dirname, join

import rastervision as rv

from .preprocess import PREPROCESS
from .constants import TRAIN_IDS, VALID_IDS


class SplitImages(rv.ExperimentSet):
    def exp_split_images(self, root_uri, train_stac_uri):
        split_dir = join(root_uri, "split_images")
        image_ids = TRAIN_IDS + VALID_IDS
        image_uris = [
            join(dirname(train_stac_uri), area, uid, "{}.tif".format(uid))
            for area, uid in image_ids
        ]

        config = (
            rv.CommandConfig.builder(PREPROCESS)
            .with_root_uri(root_uri)
            .with_config(items=image_uris, split_dir=split_dir)
            .build()
        )
        return config
