"""Train, evaluate, and use a LeNet-5 digit classifier on MNIST."""

import argparse
import random
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import matplotlib
matplotlib.use("Agg")  # Save figures on laptops and headless machines alike.
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHECKPOINT = ROOT / "models" / "lenet5.pt"
OUTPUTS = ROOT / "outputs"
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Both MNIST and prepared custom images enter here as 28 x 28 grayscale images.
PREPROCESS = transforms.Compose([
    transforms.Pad(2, fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


class LeNet5(nn.Module):
    """The classroom adaptation with dense convolutions and a linear classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = nn.AvgPool2d(2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.AvgPool2d(2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)
        self.tanh = nn.Tanh()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.s2(self.tanh(self.c1(images)))
        x = self.s4(self.tanh(self.c3(x)))
        x = self.tanh(self.c5(x)).flatten(start_dim=1)
        x = self.tanh(self.f6(x))
        return self.output(x)  # CrossEntropyLoss expects raw logits.


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mnist_loader(training: bool) -> DataLoader:
    dataset = datasets.MNIST(DATA, train=training, download=True, transform=PREPROCESS)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=training)


def save_curves(losses: list[float], accuracies: list[float]) -> None:
    epochs = range(1, len(losses) + 1)
    for values, label, filename in [
        (losses, "Mean cross-entropy", "training_loss.png"),
        (accuracies, "Accuracy (%)", "training_accuracy.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        ax.plot(epochs, values, marker="o", color="#186a75")
        ax.set(xlabel="Epoch", ylabel=label, title=f"MNIST training · {label}")
        if len(losses) <= 15:
            ax.set_xticks(list(epochs))
        else:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(OUTPUTS / filename, dpi=150)
        plt.close(fig)


def train(model: LeNet5, device: torch.device, epochs: int) -> None:
    loader = mnist_loader(training=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    losses, accuracies = [], []
    model.train()
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        loss_sum, correct, count = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            count += labels.numel()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()

        losses.append(loss_sum / count)
        accuracies.append(100 * correct / count)
        print(f"Epoch {epoch}/{epochs} | Training loss {losses[-1]:.4f} | "
              f"Training accuracy {accuracies[-1]:.2f}% | {perf_counter() - started:.1f}s",
              flush=True)

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    # Replace the previous checkpoint only after the new file is complete.
    with NamedTemporaryFile(dir=CHECKPOINT.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(model.state_dict(), temporary)
        temporary.replace(CHECKPOINT)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved weights to {CHECKPOINT}")
    OUTPUTS.mkdir(exist_ok=True)
    save_curves(losses, accuracies)


def load_model(device: torch.device) -> LeNet5:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError("Checkpoint missing. Train first with python main.py --train")
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    return model.eval()


Sample = tuple[torch.Tensor, int, int]


def save_samples(samples: list[Sample], title: str, filename: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), layout="constrained")
    fig.suptitle(title)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (image, actual, predicted) in zip(axes.flat, samples):
        ax.imshow(image.squeeze(0), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"Actual {actual} · Pred {predicted}", fontsize=10,
                     color="#186a75" if actual == predicted else "#ae362b")
    if not samples:
        fig.text(0.5, 0.5, "All test images were classified correctly.", ha="center")
    fig.savefig(OUTPUTS / filename, dpi=150)
    plt.close(fig)


def save_matrix(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    heatmap = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(heatmap, ax=ax, label="Image count", shrink=0.8)
    ax.set(xlabel="Predicted digit", ylabel="Actual digit", title="MNIST test confusion matrix",
           xticks=range(10), yticks=range(10))
    for actual, predicted in np.ndindex(matrix.shape):
        count = matrix[actual, predicted]
        ax.text(predicted, actual, str(count), ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "black", fontsize=9)
    fig.savefig(OUTPUTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def collect_samples(samples: list[Sample], images: torch.Tensor, labels: torch.Tensor,
                    predictions: torch.Tensor, indices) -> None:
    for i in indices[:10 - len(samples)]:
        samples.append((images[i], int(labels[i]), int(predictions[i])))


@torch.no_grad()
def evaluate(model: LeNet5, device: torch.device) -> None:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    matrix = np.zeros((10, 10), dtype=np.int64)
    loss_sum = 0.0
    samples, mistakes = [], []
    for images, labels in mnist_loader(training=False):
        logits = model(images.to(device))
        loss_sum += criterion(logits, labels.to(device)).item()
        predictions = logits.argmax(dim=1).cpu()
        # Count repeated (actual, predicted) pairs correctly within each batch.
        np.add.at(matrix, (labels.numpy(), predictions.numpy()), 1)
        collect_samples(samples, images, labels, predictions, range(len(labels)))
        wrong_indices = torch.where(predictions != labels)[0]
        collect_samples(mistakes, images, labels, predictions, wrong_indices)

    total, correct = int(matrix.sum()), int(matrix.trace())
    print(f"Test loss: {loss_sum / total:.4f}\nTest accuracy: {100 * correct / total:.2f}%"
          f"\nCorrect: {correct} / {total}\nIncorrect: {total - correct} / {total}")
    print("Confusion matrix (rows = actual 0–9, columns = predicted 0–9)")
    print(matrix)
    OUTPUTS.mkdir(exist_ok=True)
    save_matrix(matrix)
    save_samples(samples, "MNIST test predictions", "sample_predictions.png")
    save_samples(mistakes, "Misclassified MNIST test digits", "incorrect_predictions.png")
    print(f"Saved evaluation figures to {OUTPUTS}")


@torch.no_grad()
def demo(model: LeNet5, device: torch.device) -> None:
    images, labels = next(iter(mnist_loader(training=False)))
    predictions = model(images[:10].to(device)).argmax(dim=1).cpu()
    samples = []
    collect_samples(samples, images, labels, predictions, range(10))
    OUTPUTS.mkdir(exist_ok=True)
    save_samples(samples, "MNIST test predictions", "sample_predictions.png")
    print(f"Open {OUTPUTS / 'sample_predictions.png'}")
    for i, (_, actual, predicted) in enumerate(samples, start=1):
        print(f"Sample {i:2d} | Actual {actual} | Predicted {predicted}")


def prepare_digit(path: Path) -> torch.Tensor:
    """Extract one digit on a reasonably uniform background, then match MNIST."""
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        gray = rgba.convert("L")
        alpha = rgba.getchannel("A")
        if alpha.getextrema()[0] < 255:
            visible = np.asarray(gray)[np.asarray(alpha) > 0]
            if not visible.size:
                raise ValueError("The image is transparent. Provide a visible digit.")
            paper = 255 if np.median(visible) < 128 else 0
            gray = Image.composite(gray, Image.new("L", gray.size, paper), alpha)

    pixels = np.asarray(gray, dtype=np.float32)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    background = np.median(border)
    # Ink can be either lighter or darker than the surrounding paper.
    ink = pixels - background if background < 128 else background - pixels
    peak = ink.max()
    if peak < 8:
        raise ValueError("No clear digit found. Use one contrasting digit on a plain background.")
    ink[ink < 0.15 * peak] = 0
    digit = Image.fromarray(np.uint8(255 * ink / peak))
    digit = digit.crop(digit.getbbox())
    scale = 20 / max(digit.size)
    size = tuple(max(1, round(side * scale)) for side in digit.size)
    digit = digit.resize(size, Image.Resampling.LANCZOS)

    # Center the intensity-weighted ink mass, just as in the MNIST preparation.
    weights = np.asarray(digit, dtype=np.float32)
    y, x = np.indices(weights.shape)
    left = round(13.5 - float(np.average(x, weights=weights)))
    top = round(13.5 - float(np.average(y, weights=weights)))
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, (max(0, min(28 - digit.width, left)),
                         max(0, min(28 - digit.height, top))))
    return PREPROCESS(canvas)


@torch.no_grad()
def predict(model: LeNet5, device: torch.device, path: Path) -> None:
    image = prepare_digit(path)
    logits = model(image.unsqueeze(0).to(device))
    confidence, digit = logits.softmax(dim=1).max(dim=1)
    print(f"Predicted digit: {digit.item()}\nConfidence: {100 * confidence.item():.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--train", action="store_true", help="train from scratch, save, then evaluate")
    action.add_argument("--evaluate", action="store_true", help="score the checkpoint on the test set")
    action.add_argument("--predict", type=Path, metavar="IMAGE", help="recognize one handwritten digit")
    action.add_argument("--demo", action="store_true", help="print and plot ten test predictions")
    action.add_argument("--architecture", action="store_true", help="display the model and parameter count")
    parser.add_argument("--epochs", type=int, help="training epochs, default 5")
    args = parser.parse_args()
    if args.epochs is not None and (not args.train or args.epochs < 1):
        parser.error("--epochs requires --train and a positive integer")
    if not any((args.train, args.evaluate, args.predict, args.demo, args.architecture)):
        parser.print_help()
        return 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.architecture:
        model = LeNet5()
        print(model)
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("Input: [batch, 1, 32, 32]   Output: [batch, 10]")
        return 0

    device = select_device()
    print(f"Device: {device}", flush=True)
    try:
        if args.train:
            model = LeNet5().to(device)
            train(model, device, args.epochs or 5)
            evaluate(model, device)
        else:
            if args.predict and not args.predict.is_file():
                raise FileNotFoundError(f"Image file not found: {args.predict}")
            model = load_model(device)
            if args.predict:
                predict(model, device, args.predict)
            elif args.evaluate:
                evaluate(model, device)
            else:
                demo(model, device)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
