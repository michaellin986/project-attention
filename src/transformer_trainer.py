import time
import torch


class TransformerTrainer:
    def __init__(
        self,
        optimizer,
        model,
        loss_func,
        en_tokenizer,
        xx_tokenizer,
        **kwargs,
    ):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = kwargs["d_model"]
        self.optimizer = optimizer(model.parameters(), **kwargs)
        self.model = model.to(self.device)
        self.loss_func = loss_func
        self.en_tokenizer = en_tokenizer
        self.xx_tokenizer = xx_tokenizer
        self.epoch = 0
        self.start_time = None

        # Added for testing
        self.en_embedding = torch.nn.Embedding(len(self.en_tokenizer), self.d_model)
        self.xx_embedding = torch.nn.Embedding(len(self.xx_tokenizer), self.d_model)
        self.emb_scale = self.d_model**0.5
        self.relu = torch.nn.ReLU()

    def undo_embedding(self, embedded):
        # Undo the embedding
        reconstructed = torch.matmul(
            embedded.to(self.device), self.en_embedding.weight.t().to(self.device)
        ).to(self.device)

        # Return reconstructed input indices
        return torch.argmax(reconstructed, dim=1).to(self.device)

    def run_one_batch(self, x, y, train=True, display=False):
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
        else:
            self.model.eval()

        emb_x, emb_y = self.xx_embedding(x) * self.emb_scale, self.en_embedding(y)
        outputs = self.model(emb_x.to(self.device), emb_y.to(self.device)).to(
            self.device
        )

        # Swap axes so that both `outputs` and `targets` have shape
        # [batch_size, en_vocab_size, embedding_size], as required
        # by torch.nn.CrossEntropyLoss:
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss = self.loss_func(
            torch.permute(outputs, (0, 2, 1)),
            torch.permute(emb_y, (0, 2, 1)).to(self.device),
        ).to(self.device)

        if train:
            loss.backward()
            self.optimizer.step()

        if display:
            print("Input text (French/German):")
            print(self.xx_tokenizer.decode(x[0]))
            target, output = self.undo_embedding(emb_y[0]), self.undo_embedding(
                outputs[0]
            )
            print("\nTarget translation text (English)")
            print(self.en_tokenizer.decode(target.int()))
            print("\nModel's translation text (English)")
            print(self.en_tokenizer.decode(self.relu(torch.round(output)).int()))
            print("\n\n")

        return loss.detach()

    def run_one_epoch(self, data_loader, train=True, verbose=True):
        if self.start_time is None:
            self.start_time = time.time()
        epoch_size = 0
        total_loss = 0
        for i, (x, y) in enumerate(data_loader):
            epoch_size += x.size(0)
            loss = self.run_one_batch(x, y, train=train, display=i == 0)
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
                    f"Average Loss: {avg_loss:6.3f}",
                    f"in {duration:5.1f} min",
                ]
            )
            print("  ".join(log))

        return avg_loss

    def train(self, data_loader, n_epochs, folder_name=None, train=True):
        losses = []
        for _ in range(n_epochs):
            loss = self.run_one_epoch(data_loader, train=train)
            losses.append(loss)
            if train:
                self.epoch += 1
                if folder_name and (self.epoch == n_epochs or self.epoch % 100 == 0):
                    torch.save(
                        self.model, f"./models/{folder_name}/epoch_{self.epoch}.pt"
                    )

        return losses
