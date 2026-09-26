import h5py
import matplotlib.pyplot as plt
import numpy as np

def print_h5_structure(g, indent=0):
    """Rekursiv alle Gruppen und Datasets auflisten"""
    for key in g.keys():
        item = g[key]
        if isinstance(item, h5py.Group):
            print("  " * indent + f"[Group] {key}")
            print_h5_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            data = item[()]
            print("  " * indent + f"[Dataset] {key}: shape={data.shape}, dtype={data.dtype}")
            print("  " * indent + f"  Werte: {data.ravel()[:5]} ...")


def get_weights(f, idx):

    return f["layers"]["dense"]["vars"][idx][()]
    #bias = f["dense"]["dense"]["bias:0"][()]


if __name__ == '__main__':
    # Datei öffnen
    f = h5py.File("out/ex_02.weights.h5", "r")
    print_h5_structure(f)
    w0 = get_weights(f, "0")


    for neuron_idx in range(w0.shape[1]):
        weights_img = w0[:, neuron_idx].reshape(28, 28)

        plt.imshow(weights_img, cmap="seismic")
        plt.colorbar(label="Gewichtswert")
        plt.title(f"Neuron {neuron_idx} - Gewichtsmuster")
        plt.savefig(f"out/weights/ex_02.neuron{neuron_idx}.weights.png")
        plt.close()



    w1 = get_weights(f, "1")
