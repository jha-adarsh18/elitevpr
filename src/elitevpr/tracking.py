"""Run logging. `logging=none` keeps tests and public clones account-free."""


class NullLogger:
    run_id = None

    def log(self, metrics):
        pass

    def finish(self):
        pass


class WandbLogger:
    def __init__(self, project, config, run_id=None, mode=None):
        import wandb

        self._wandb = wandb
        kwargs = {"project": project, "config": config, "resume": "allow"}
        if run_id:
            kwargs["id"] = run_id
        if mode:
            kwargs["mode"] = mode
        wandb.init(**kwargs)
        self.run_id = wandb.run.id

    def log(self, metrics):
        self._wandb.log(metrics)

    def finish(self):
        self._wandb.finish()


def build_logger(cfg, config, run_id=None):
    name = str(cfg.name)
    if name == "none":
        return NullLogger()
    if name == "wandb":
        return WandbLogger(project=cfg.project, config=config, run_id=run_id,
                           mode=cfg.get("mode"))
    raise ValueError(f"unknown logger {name!r}")
