# LeNet-5 for handwritten digit recognition

Give the program an image of a single handwritten digit and it returns a digit from 0 to 9. The same program can train the classifier from scratch on MNIST, evaluate it on 10,000 held-out images, and show which digits it confused. All Python code lives in `main.py`.

## The research paper

In 1998, Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner studied how a machine could read handwriting despite variations in stroke width, position, and writing style. A practical application was reading amounts on bank checks. The authors compared recognition methods on isolated digits and described how to train components of a larger document reader together. This project implements the isolated-digit task. [Paper and citation from coauthor Léon Bottou](https://leon.bottou.org/papers/lecun-98h)

A convolutional network learns filters directly from examples. Each filter examines a small neighborhood and uses the same weights across the image, so a stroke detector can respond at many positions. Later layers combine those responses into larger shapes. Gradients carry information about a classification error back to all these filters. LeNet-5 helped establish that a network could learn both visual features and a classifier from labeled pixels, a design still used in computer vision. [1998 paper, sections I–III](https://princetonuniversity.github.io/NEU-PSY-502/_static/pdf/Class%204/LeCunn1998.pdf)

## Architecture

Shapes below omit the batch dimension and use channels × height × width. Every convolution has stride 1, a learned bias, and zero internal padding. A 5 × 5 filter on a 32 × 32 image fits in 28 × 28 positions because 32 − 5 + 1 = 28.

| Stage | Input | Output | What the layer does |
|---|---|---|---|
| C1 | 1 × 32 × 32 | 6 × 28 × 28 | Six learned 5 × 5 filters look for local patterns such as edges. Tanh transforms each response. |
| S2 | 6 × 28 × 28 | 6 × 14 × 14 | Average pooling replaces each non-overlapping 2 × 2 patch with its mean. This preserves coarse location and reduces sensitivity to small shifts. |
| C3 | 6 × 14 × 14 | 16 × 10 × 10 | Each 5 × 5 filter combines all six incoming maps. Tanh allows combinations of simpler features to encode shapes such as curves. |
| S4 | 16 × 10 × 10 | 16 × 5 × 5 | A second 2 × 2 average pool halves both spatial dimensions. |
| C5 | 16 × 5 × 5 | 120 × 1 × 1 | Each filter spans the entire remaining field. Its tanh response summarizes one learned feature of the digit. |
| Flatten | 120 × 1 × 1 | 120 | Reshapes each example into a vector while preserving its values and the batch dimension. |
| F6 | 120 | 84 | Each neuron learns a weighted combination of all 120 features, adds a bias, and applies tanh. |
| Output | 84 | 10 | A linear layer gives one raw score (logit) per digit. The largest score selects the prediction. |

There are **61,706 trainable parameters**. For example, C1 uses 6 × (1 × 5 × 5 + 1) = 156. The same 156 parameters operate at every spatial position. Run `python main.py --architecture` to inspect the modules and count their parameters.

## How gradient-based learning works

1. **Forward propagation.** A batch of images passes through the network. Each image receives ten logits, denoted by $z_0,\ldots,z_9$.
2. **Loss.** For an image whose true digit is $y$, cross-entropy is $L = \log(\sum_{j=0}^{9} e^{z_j}) - z_y$. Increasing the true digit's score relative to the other scores reduces the loss. PyTorch's `CrossEntropyLoss` evaluates this from logits and averages the batch.
3. **Backpropagation.** `loss.backward()` applies the chain rule from the loss back through F6 and the convolutional filters. A gradient $g = \partial L/\partial w$ estimates how much the loss changes for a small change to one weight $w$.
4. **Weight update.** Ordinary gradient descent uses $w \leftarrow w - \eta g$, where $\eta$ is the learning rate. Adam tracks moving averages of gradients and squared gradients. Its update is $w \leftarrow w - \eta\hat m/(\sqrt{\hat v}+\epsilon)$, using bias-corrected averages and a small constant for stability.
5. **Repeat.** `optimizer.zero_grad()` clears gradients before the next forward pass. One epoch visits every training image once. The defaults are five epochs, 64 images per batch, and Adam with a learning rate of 0.001.

## Dataset and preprocessing

Torchvision downloads MNIST into `data/` on first use and reuses those files. The training split has 60,000 grayscale 28 × 28 images. The test split has 10,000, with labels 0–9 in both. Only the training loader shuffles and supplies images for optimizer updates. The test pass uses frozen weights, evaluation mode, and disabled gradient tracking. [Torchvision MNIST documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)

MNIST already places light strokes on a dark background. Each side receives a two-pixel black border, bringing the canvas to 32 × 32 without stretching the digit. `ToTensor` maps bytes to $[0,1]$, then $x'=(x-0.5)/0.5$ maps black to −1 and white to +1. This fixed scale gives tanh both negative and positive inputs. Training, testing, and prediction share this transform.

For a custom image, use one digit with a margin of plain background. Pillow converts the image to grayscale and handles transparency. The program estimates the background from the median border brightness, makes the ink bright, and removes faint background variation. It crops to the strokes, scales the longest side to 20 pixels while preserving aspect ratio, and places the intensity-weighted center near the middle of a 28 × 28 canvas. The shared transform then adds the border and normalizes the pixels.

## Installation

Use Python 3.10 or newer with a compatible PyTorch build. Replace the repository placeholder with your clone URL. If you have this folder already, open a terminal in the directory containing `main.py` and start with the virtual environment.

```bash
git clone <repository-url> LeNet5-Document-Recognition
cd LeNet5-Document-Recognition
```

**macOS and Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --train
```

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --train
```

In Windows Command Prompt, use `.venv\Scripts\activate.bat` for activation. CUDA takes priority when PyTorch can access an NVIDIA GPU, followed by Apple Silicon MPS and then CPU. For an NVIDIA installation, choose the PyTorch build compatible with your system. [PyTorch installation selector](https://pytorch.org/get-started/locally/)

## Running the project

| Command | Result |
|---|---|
| `python main.py` or `python main.py --help` | Print usage. |
| `python main.py --architecture` | Inspect the layers and parameter count without downloading data. |
| `python main.py --train` | Train for five epochs from newly initialized weights, save the checkpoint, and evaluate the full test split. |
| `python main.py --train --epochs 10` | Run the same pipeline for ten epochs. |
| `python main.py --evaluate` | Load the saved weights, report test metrics, and regenerate the evaluation plots. |
| `python main.py --demo` | Print actual and predicted labels for the first ten test images and save their labeled grid. |
| `python main.py --predict "my digit.png"` | Load the saved model and print a digit and its softmax confidence. |

Training replaces `models/lenet5.pt` and the plot files. Data, weights, and outputs are resolved relative to `main.py`; the image path is relative to the terminal's working directory unless you give an absolute path. A checkpoint is required for evaluation, demo, and prediction. Once you have the weights, predicting your own image works offline.

Confidence is the probability the model assigns to its chosen digit after softmax. Even a high value can accompany an incorrect answer, especially on unfamiliar images. Accuracy measures the fraction of labeled examples the model actually gets right.

## Model results and plots

The saved checkpoint scored **98.34% accuracy** on the full MNIST test split. It classified **9,834** images correctly and **166** incorrectly, with mean test cross-entropy **0.0504**. This run used the defaults on Apple Silicon MPS with PyTorch 2.14.1 and torchvision 0.29.2.

| Epoch | Mean training loss | Training accuracy |
|---:|---:|---:|
| 1 | 0.2713 | 92.21% |
| 2 | 0.0854 | 97.38% |
| 3 | 0.0582 | 98.21% |
| 4 | 0.0458 | 98.58% |
| 5 | 0.0372 | 98.86% |

The most frequent error was reading **4 as 9**. Of the 982 handwritten fours, the model labeled 954 correctly and called 19 of them nines. The reverse error occurred eight times. The confusion matrix shows each direction separately.

Training loss is the mean cross-entropy per training image, and training accuracy counts correct predictions as weights change during the epoch. Test loss and accuracy use the final weights throughout. Both losses weight every image equally, including those in a shorter final batch. The program prints the number correct, the number incorrect, and the complete test confusion matrix.

| File under `outputs/` | How to read it |
|---|---|
| `training_loss.png` | Each point is one epoch. A falling curve means the model assigns more probability to correct labels on average. |
| `training_accuracy.png` | Tracks the percentage of training images whose largest logit corresponds to the label. |
| `confusion_matrix.png` | Rows are true digits and columns are predictions. The diagonal counts correct answers. Row 4, column 9 counts handwritten fours read as nines. |
| `sample_predictions.png` | The first ten test images with both labels. Matching labels appear in green. |
| `incorrect_predictions.png` | Up to ten errors in test order. Inspect their strokes for ambiguity, then consult the matrix to see how frequent that confusion is. |

## Research reproduction notes

This implementation preserves the sequence of stages and their dimensions. The following choices define the teaching adaptation. [Historical architecture and training procedure, sections II–III and appendix C](https://princetonuniversity.github.io/NEU-PSY-502/_static/pdf/Class%204/LeCunn1998.pdf)

| Component | LeNet-5 in the paper | This project |
|---|---|---|
| C3 | Selected subsets of S2 maps fed individual filters | Every filter uses every S2 map. |
| S2 and S4 | Each map learned a subsampling coefficient and bias, followed by a nonlinearity | Fixed 2 × 2 average pooling. |
| Activations | Scaled hyperbolic tangent | Standard PyTorch tanh after C1, C3, C5, and F6. |
| Classification | Euclidean RBF units scored distances to class prototypes | A learned linear head produces ten logits for cross-entropy. |
| Optimization | Stochastic diagonal Levenberg–Marquardt updates with a scheduled learning rate | Mini-batch Adam with a fixed base learning rate. |
| Inputs and experiments | The paper describes specific foreground/background levels and experiments with distorted examples | Padded, centered MNIST scaled to [−1, +1], without augmentation, for five epochs by default. |

Python, NumPy, and PyTorch are seeded with 42. Results can vary across library versions, devices, and numerical kernels.

## Limitations and future improvements

The classifier emits one label per image, so a string such as “42” needs a segmentation or sequence-recognition step. Rotation can change the identity of a digit. Clutter or shadows can spoil the border estimate and crop, while faint scans can lose strokes during resizing. Unusual handwriting may also differ substantially from MNIST, even when the crop looks clean.

Possible extensions include small affine transformations of training images to study rotation tolerance, a segmentation stage for separated digits, and a sequence recognizer for document text. A separate validation split from the training data would support tuning such experiments while reserving the test set for final reporting.

## Viva and interview preparation

1. **What does LeNet-5 learn?** It learns local visual features and a decision rule that maps a digit image to one of ten classes.
2. **Why use convolution on handwriting?** Nearby pixels form strokes, and the same stroke can occur at different locations. Shared filters exploit both properties.
3. **What is a feature map?** It is the grid of responses from one filter at all valid image locations.
4. **Why pad MNIST to 32 × 32?** Padding preserves the original digit scale and gives the sequence of convolutions the intended spatial dimensions.
5. **Why does C1 output 28 × 28?** A 5 × 5 window has 28 valid starting positions along each axis of a 32 × 32 image at stride 1.
6. **What does average pooling do?** Averages each local window and reduces spatial resolution. Each S2 or S4 output summarizes four inputs.
7. **Why use tanh?** Its curved response lets successive layers learn nonlinear relationships. Its output is bounded between −1 and +1.
8. **Why does C5 produce 1 × 1 maps?** Its 5 × 5 kernels cover the entire 5 × 5 input field, so there is only one valid position.
9. **What does flattening learn?** It has no parameters. It reshapes the 120 C5 values for each image into F6's input vector.
10. **What is a logit?** It is an unnormalized class score. Taking the index of the largest logit yields the predicted digit.
11. **Why feed logits into cross-entropy?** The loss computes log-softmax internally, which evaluates the log probabilities stably.
12. **How does backpropagation differ from an optimizer?** Backpropagation computes gradients with the chain rule. The optimizer uses them to change parameters.
13. **What does Adam remember?** Running estimates of the gradient and its square help set the direction and size of each parameter update.
14. **Why keep the test split separate?** It measures performance on images excluded from fitting. Tuning repeatedly on test results would bias that estimate.
15. **How can two models have equal accuracy and different losses?** Accuracy uses only the winning class. Cross-entropy also reflects how much probability each model assigned to the true class.

## Citation

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. “Gradient-Based Learning Applied to Document Recognition.” *Proceedings of the IEEE*, 86(11), 2278–2324, November 1998. [DOI 10.1109/5.726791](https://doi.org/10.1109/5.726791) · [Coauthor's paper archive](https://leon.bottou.org/papers/lecun-98h)
