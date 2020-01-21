from os.path import dirname, join

import rastervision as rv

from .preprocess import PREPROCESS
from .constants import ALL_T1_IDS


class SplitImages(rv.ExperimentSet):
    def exp_split(self, root_uri, train_stac_uri, split_dir):
        image_ids = ALL_T1_IDS
        image_uris = [
            join(dirname(train_stac_uri), area, uid, f"{uid}.tif")
            for area, uid in image_ids
        ]

        config = (
            rv.CommandConfig.builder(PREPROCESS)
            .with_root_uri(root_uri)
            .with_config(items=image_uris, split_dir=split_dir)
            .build()
        )
        return config
