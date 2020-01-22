import rastervision as rv

NOOP = "NOOP"


class NoOp(rv.ExperimentSet):
    def exp_noop(self):
        root_uri = "s3://carderne-rv/"
        config = (
            rv.CommandConfig.builder(NOOP)
            .with_root_uri(root_uri)
            .with_config(root_uri=root_uri)
            .build()
        )
        return config


class NoOpCommand(rv.AuxCommand):
    command_type = NOOP
    options = rv.AuxCommandOptions(
        inputs=lambda conf: NoOpCommand.gather_inputs(conf),
        outputs=lambda conf: NoOpCommand.gather_outputs(conf),
        required_fields=["root_uri"],
    )

    def run(self):
        print("noop working")

    @staticmethod
    def gather_inputs(conf):
        return conf["root_uri"]

    @staticmethod
    def gather_outputs(conf):
        return [".phony"]


def register_plugin(plugin_registry):
    plugin_registry.register_aux_command(NOOP, NoOpCommand)
