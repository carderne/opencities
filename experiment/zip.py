import os
from os.path import join
from subprocess import call

import rastervision as rv
from rastervision.utils.files import upload_or_copy, zipdir

ZIP = "ZIP"


class ZipProcessed(rv.ExperimentSet):
    def exp_zip(self, experiment_id, root_uri, test_stac_uri):
        config = (
            rv.CommandConfig.builder(ZIP)
            .with_root_uri(root_uri)
            .with_config(experiment_id=experiment_id, root_uri=root_uri)
            .build()
        )
        return config


class ZipProcessedCommand(rv.AuxCommand):
    command_type = ZIP
    options = rv.AuxCommandOptions(
        inputs=lambda conf: ZipProcessedCommand.gather_inputs(conf),
        outputs=lambda conf: ZipProcessedCommand.gather_outputs(conf),
        required_fields=["experiment_id", "root_uri"],
    )

    def run(self):
        experiment_id = self.command_config.get("experiment_id")
        root_uri = self.command_config.get("root_uri")
        tmp_dir = "/opt/data/tmp/"
        os.makedirs(tmp_dir)
        print("made tmp_dir")
        s3_command = (
            f"aws s3 sync s3://carderne-rv/postprocess/{experiment_id}/ {tmp_dir}"
        )
        call(s3_command, shell=True)
        print("downloaded files")
        zip_file = f"/opt/data/zipped.zip"
        zipdir(tmp_dir, zip_file)
        print("zipped")
        upload_or_copy(zip_file, join(root_uri, "final", f"{experiment_id}.zip"))
        print("uploaded")

    @staticmethod
    def gather_inputs(conf):
        return conf["experiment_id"]

    @staticmethod
    def gather_outputs(conf):
        return [".phony"]


def register_plugin(plugin_registry):
    plugin_registry.register_aux_command(ZIP, ZipProcessedCommand)
