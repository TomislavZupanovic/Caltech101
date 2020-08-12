import matplotlib.pyplot as plt


def plot_losses(losses):
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, squeeze=True,
                            figsize=(3, 3))
    for loss in ['train_loss', 'valid_loss']:
        axs[0, 0].plot(losses[loss], label=loss)
    axs[0, 0].legend(loc='best')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Average Loss')
    axs[0, 0].set_title('Training and validation loss')
    axs[0, 0].grid()
    for acc in ['train_acc', 'valid_acc']:
        axs[0, 1].plot(losses[loss] * 100, label=acc)
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Average Accuracy')
    axs[0, 1].set_title('Training and Validation Accuracy')
    plt.tight_layout()
    plt.show()

