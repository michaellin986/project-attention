import time


class Trainer:
    def __init__(self, optimizer, model, loss_func, **kwargs):
        self.optimizer = optimizer(model.parameters(), **kwargs)
        self.model = model
        self.loss_func = loss_func
        self.epoch = 0
        self.start_time = None

    def run_one_batch(self, x, y, train=True):
        if train:
            self.model.train()
            self.optimizer.zero_grad()

        output = self.model(x, x, x)
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
            epoch_size += x.size(0)
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
