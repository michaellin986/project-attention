import time


class Trainer:
    def __init__(self, optimizer, model, loss_func, **kwargs):
        """
        Generic trainer for testing purposes
        Common kwargs include those used by chosen optimizer:
        - lr: float
            - doesn't matter when using DynamicLRAdam optimizer because overridden
        - betas: Tuple[float, float]
            - paper sets this to (0.9, 0.98)
        - eps: float
            - paper sets this to 10e-9
        - d_model: int
            - only when using DynamicLRAdam optimizer; paper sets this to 512
        - warmup_steps: int
            - only when using DynamicLRAdam optimizer; paper sets this to 4000
        """
        self.optimizer = optimizer(model.parameters(), **kwargs)
        self.model = model
        self.loss_func = loss_func
        self.epoch = 0
        self.start_time = None

    def run_one_batch(self, x, y, train=True):
        """
        x: a tuple of row tensor (this is so that modules
        like Multihead Attention, which takes 3 inputs,
        can be tested with the same data structure).

        y: a single row tensor.

        train: boolean
        """
        if train:
            self.model.train()
            self.optimizer.zero_grad()

        output = self.model(*x)
        loss = self.loss_func(output, y)

        if train:
            loss.backward()
            self.optimizer.step()

        return loss.detach()

    def run_one_epoch(self, data_loader, train=True, verbose=True):
        if self.start_time is None:
            self.start_time = time.time()
        epoch_size = 0
        total_loss = 0
        for batch_data in data_loader:
            x, y = batch_data
            epoch_size += x[0].size(0)
            loss = self.run_one_batch(x, y, train=train)
            total_loss += loss

        avg_loss = total_loss / epoch_size

        if verbose:
            epoch = self.epoch + 1
            duration = (time.time() - self.start_time) / 60

            if train:
                log = [f"Epoch: {epoch:6d}"]
            else:
                log = ["Eval:" + " " * 8]

            log.extend(
                [
                    f"Loss: {avg_loss:6.3f}",
                    f"in {duration:5.1f} min",
                ]
            )
            print("  ".join(log))

        return avg_loss

    def train(self, data_loader, n_epochs, train=True):
        losses = []
        for _ in range(n_epochs):
            loss = self.run_one_epoch(data_loader, train=train)
            losses.append(loss)
            if train:
                self.epoch += 1

        return losses
