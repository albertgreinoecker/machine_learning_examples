"""
Visualize what the neurons of the simple MNIST network from
ex_02_first_learning.py have learned.

This script:
  1. loads the trained model from ``out/ex_02_model.keras`` (or trains the same
     784 -> 24 -> 16 -> 10 network if no saved model exists),
  2. shows the weights of each neuron of the first hidden layer as a 28x28
     image (red = pixel excites the neuron, blue = pixel inhibits it),
  3. shows "effective templates" for the deeper layers by multiplying the
     weight matrices (W1 @ W2, W1 @ W2 @ W3, ...). This is a linear approximation
     that ignores the ReLU activations,
  4. compares the effective templates of the output layer with the mean
     image of each digit (computed on the training set),
  5. shows a heatmap "neuron x digit class" with the mean activation of each
     neuron for each digit 0-9 (computed on the test set),
  6. uses activation maximization: starting from an almost empty image,
     gradient ascent changes the pixels so that a single neuron becomes as
     active as possible. The result is the input the neuron "likes" most.

Pixels at the border that (almost) never contain ink in the training set are
masked out (shown white) in the weight images: their weights never receive a
gradient during training and therefore keep their random initial values.

All figures are saved as PNG files to ``out/ex_02_visualization/``.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Do not use the GPU
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras.src.datasets import mnist

MODEL_PATH = "out/ex_02_model.keras"
OUTPUT_DIR = "out/ex_02_visualization/"
# pixels with ink in less than this fraction of training images are masked out
MIN_INK_FRACTION = 0.01
# activation maximization: number of gradient steps, learning rate, L2 penalty
AM_STEPS = 300
AM_LEARNING_RATE = 0.05
AM_L2_PENALTY = 0.05
num_classes = 10

"""
## Prepare the data (same preprocessing as in ex_02_first_learning.py)
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

# 1 for pixels that are inked often enough to be learned, 0 otherwise -> (784,)
pixel_mask = ((x_train > 0).mean(axis=0) >= MIN_INK_FRACTION).astype("float32")
print(int(784 - pixel_mask.sum()), "of 784 pixels are masked out (almost never inked)")

"""
## Load the trained model, or train it if it does not exist yet
"""

if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
else:
    model = keras.Sequential(
        [
            keras.Input(shape=(784,)),
            layers.Dense(24, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(
        x_train,
        keras.utils.to_categorical(y_train, num_classes),
        batch_size=128,
        epochs=17,
        validation_split=0.1,
    )
    os.makedirs("out", exist_ok=True)
    model.save(MODEL_PATH)

model.summary()

# get_weights() returns [kernel, bias]; kernel has shape (inputs, neurons),
# e.g. (784, 24), (24, 16), (16, 10). Works for any number of Dense layers.
kernels = [layer.get_weights()[0] for layer in model.layers]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, filename):
    """Save the figure to OUTPUT_DIR and free its memory."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def plot_templates(weights, title, filename, n_cols, labels=None, diverging=True):
    """Plot each column of a (784, n) matrix as a 28x28 image.

    diverging=True: weights, red/blue around 0. diverging=False: pixel values in [0, 1].
    """
    # set the weights of (almost) never inked pixels to 0 (white)
    weights = weights * pixel_mask[:, None]
    n = weights.shape[1]
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.8 * n_rows))
    for i, ax in enumerate(np.ravel(axes)):
        ax.axis("off")
        if i >= n:
            continue
        img = weights[:, i].reshape(28, 28)
        if diverging:
            # symmetric color range so that 0 is white in the diverging colormap
            limit = np.abs(img).max()
            ax.imshow(img, cmap="bwr", vmin=-limit, vmax=limit)
        else:
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(labels[i] if labels else f"Neuron {i}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, filename)


"""
## 1. Weights of the first hidden layer as images
Each of the 24 neurons has exactly one weight per pixel.
"""

plot_templates(
    kernels[0],
    "Layer 1: weights per neuron (red = excite, blue = inhibit)",
    "layer1_weights.png",
    n_cols=6,
)

"""
## 2. Effective templates of the deeper layers
Linear approximation: multiply the weight matrices, ignoring ReLU and biases.
"""

effective = kernels[0]
for layer_idx, kernel in enumerate(kernels[1:], start=2):
    effective = effective @ kernel  # (784, neurons of this layer)
    is_output = layer_idx == len(kernels)
    plot_templates(
        effective,
        f"{'Output layer' if is_output else f'Layer {layer_idx}'}: "
        f"effective templates (W1 @ ... @ W{layer_idx})",
        f"layer{layer_idx}_effective_templates.png",
        n_cols=5 if is_output else 8,
        labels=[f"Digit {d}" for d in range(num_classes)] if is_output else None,
    )

"""
## 3. Output templates next to the mean image of each digit
After the loop above, ``effective`` holds the templates of the output layer.
"""

fig, axes = plt.subplots(2, num_classes, figsize=(1.6 * num_classes, 3.8))
for d in range(num_classes):
    mean_img = x_train[y_train == d].mean(axis=0).reshape(28, 28)
    axes[0, d].imshow(mean_img, cmap="gray_r", vmin=0, vmax=1)
    axes[0, d].set_title(f"Digit {d}", fontsize=8)

    template = (effective[:, d] * pixel_mask).reshape(28, 28)
    limit = np.abs(template).max()
    axes[1, d].imshow(template, cmap="bwr", vmin=-limit, vmax=limit)
for ax in np.ravel(axes):
    ax.axis("off")
fig.suptitle("Top: mean image per digit (training set) / bottom: effective output template")
fig.tight_layout()
save_figure(fig, "mean_image_vs_output_template.png")

"""
## 4. Heatmap: mean activation per neuron and digit class
A helper model returns the outputs of all layers at once.
"""

activation_model = keras.Model(
    inputs=model.inputs, outputs=[layer.output for layer in model.layers]
)
activations = activation_model.predict(x_test, verbose=0)

fig, axes = plt.subplots(1, len(activations), figsize=(5 * len(activations), 6))
for layer_idx, (ax, act) in enumerate(zip(axes, activations)):
    # mean activation of every neuron, grouped by the true label -> (neurons, 10)
    mean_act = np.stack([act[y_test == d].mean(axis=0) for d in range(num_classes)], axis=1)
    im = ax.imshow(mean_act, cmap="viridis", aspect="auto")
    ax.set_xticks(range(num_classes))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Neuron")
    ax.set_title(f"Layer {layer_idx + 1} ({act.shape[1]} neurons)")
    fig.colorbar(im, ax=ax)
fig.suptitle("Mean activation per neuron and digit class (test set)")
fig.tight_layout()
save_figure(fig, "mean_activation_per_class.png")

"""
## 5. Activation maximization
For every neuron, find an input image that makes this neuron as active as
possible: start with an almost empty image and follow the gradient of the
neuron's value with respect to the pixels (gradient ascent).

- We maximize the value *before* the activation function (pre-activation).
  ReLU would give a gradient of 0 for an inactive neuron, and softmax would
  let the output neurons compete with each other.
- An L2 penalty keeps the image from simply becoming as bright as possible.
- Pixels are clipped to [0, 1] and the masked border pixels are kept at 0, so
  the result stays a valid MNIST-like image.
- All neurons of a layer are optimized at once: image i belongs to neuron i.
"""


def pre_activations(x):
    """Forward pass that returns the values before the activation of every layer."""
    outputs = []
    h = x
    for layer in model.layers:
        z = tf.matmul(h, layer.kernel) + layer.bias
        outputs.append(z)
        h = layer.activation(z)
    return outputs


mask = tf.constant(pixel_mask)
for layer_idx, layer in enumerate(model.layers):
    n_neurons = layer.units
    rng = np.random.default_rng(42)
    images = tf.Variable(rng.uniform(0.0, 0.1, size=(n_neurons, 784)).astype("float32"))
    optimizer = keras.optimizers.Adam(learning_rate=AM_LEARNING_RATE)

    for _ in range(AM_STEPS):
        with tf.GradientTape() as tape:
            z = pre_activations(images * mask)[layer_idx]  # (n_neurons, n_neurons)
            # image i should activate neuron i -> take the diagonal
            neuron_values = tf.linalg.diag_part(z)
            l2 = tf.reduce_sum(tf.square(images * mask), axis=1)
            loss = -tf.reduce_sum(neuron_values - AM_L2_PENALTY * l2)
        grads = tape.gradient(loss, [images])
        optimizer.apply_gradients(zip(grads, [images]))
        images.assign(tf.clip_by_value(images, 0.0, 1.0))

    result = (images * mask).numpy()
    is_output = layer_idx == len(model.layers) - 1
    plot_templates(
        result.T,
        f"{'Output layer' if is_output else f'Layer {layer_idx + 1}'}: "
        "activation maximization (input that excites the neuron most)",
        f"layer{layer_idx + 1}_activation_maximization.png",
        n_cols=5 if is_output else 8,
        labels=[f"Digit {d}" for d in range(num_classes)] if is_output else None,
        diverging=False,
    )
