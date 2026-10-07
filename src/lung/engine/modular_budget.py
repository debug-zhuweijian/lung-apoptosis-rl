"""A resumable stage pause is distinct from scientific training completion."""


class StagePaused(RuntimeError):
    def __init__(self, reason, checkpoint_path=None, *, stage=None, epoch=0, step=0,
                 cursor=0):
        super().__init__(str(reason))
        self.reason = str(reason)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.stage = stage
        self.epoch = int(epoch)
        self.step = int(step)
        self.cursor = int(cursor)
