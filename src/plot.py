import json
import matplotlib.pyplot as plt


def plot_loss_and_bleu_scores(data_file_name, save_file_name=None):
    with open(data_file_name) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Loss and BLEU Score")
    fig.set_size_inches(8, 4)
    ax1.plot(data["train_losses"], label="Train")
    ax1.plot(data["validation_losses"], label="Validation")
    ax1.set(xlabel="Epoch", ylabel="Loss")
    ax1.legend()

    ax2.plot(data["train_bleu_scores"], label="Train")
    ax2.plot(data["validation_bleu_scores"], label="Validation")
    ax2.set(xlabel="Epoch", ylabel="BLEU Score")
    ax2.legend()

    if save_file_name is not None:
        fig.savefig(save_file_name)


if __name__ == "__main__":
    plot_loss_and_bleu_scores(
        "models/20230608_022058_fr-en_N1000_B100_L50_E25/epoch_25_losses.json",
        "models/20230608_022058_fr-en_N1000_B100_L50_E25/epoch_25_plots.png",
    )
    plot_loss_and_bleu_scores(
        "models/20230608_024909_de-en_N1000_B100_L50_E25/epoch_25_losses.json",
        "models/20230608_024909_de-en_N1000_B100_L50_E25/epoch_25_plots.png",
    )
    plot_loss_and_bleu_scores(
        "models/20230608_015040_fr-en_N1000_B100_L50_E25_MSE/epoch_25_losses.json",
        "models/20230608_015040_fr-en_N1000_B100_L50_E25_MSE/epoch_25_plots.png",
    )
