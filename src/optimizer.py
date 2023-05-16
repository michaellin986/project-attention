import torch


class DynamicLRAdam(torch.optim.Adam):
    """
    Use this like any other optimizer from torch, with
    the additional parameters:
    - d_model: paper sets this to 512
    - warmup_steps: paper sets this to 4000
    """

    def __init__(self, *args, **kwargs):
        self.d_model = kwargs.pop("d_model")
        self.warmup_steps = kwargs.pop("warmup_steps")
        self.step_num = 0

        if self.d_model is None:
            raise TypeError("Expects `d_model` kwarg but received None")
        if self.warmup_steps is None:
            raise TypeError("Expects `warmup_steps` kwarg but received None")

        super().__init__(*args, **kwargs)

    def compute_lr(self):
        return self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5),
        )

    def step(self, closure=None):
        self.step_num += 1
        lr = self.compute_lr()
        for group in self.param_groups:
            group["lr"] = lr
        super().step(closure)
