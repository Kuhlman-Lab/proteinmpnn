import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


def plot_probs(data, args, png_name):

    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    fig=plt.figure()
    ax = plt.axes()
    im = ax.imshow(data, cmap=args.cmap)
    # Add labels and format them
    ax.set_xlabel('Position')
    ax.set_ylabel('Residue')
    ax.set_xticks(np.arange(data.shape[1]), labels=np.arange(data.shape[1]) + 1)
    ax.set_yticks(np.arange(data.shape[0]), labels=alphabet)
    ax.tick_params(axis='x', which='major', labelsize=6, labelrotation=90)
    ax.tick_params(axis='y', which='major', labelsize=6)
    # Add colorbar
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    # Save to disk
    plt.savefig(png_name, dpi=400)

    return


def load_file(f):
    assert f.endswith('.npz'), print("Files must end with .npz")
    with open(f, 'rb') as fopen:
        data = np.load(fopen).T
    return data

def main(args):
    for f in args.input:
        data = load_file(f)
        png_name = f.removesuffix('.npz') + '.png'
        plot_probs(data, args, png_name)

    return


def subtract(args):

    probs1 = load_file(args.input[0])
    probs2 = load_file(args.input[1])
    
    if probs1.shape != probs2.shape:
        raise ValueError(f"Probabilities with shape {probs1.shape} don't match those with shape {probs2.shape}")

    diff = probs1 - probs2
    out_dir = os.path.dirname(args.input[0])
    f1 = os.path.basename(args.input[0]).removesuffix('.npz')
    f2 = os.path.basename(args.input[1]).removesuffix('.npz')
    png_name = os.path.join(
        out_dir, f1 + '_minus_' + f2 + '.png'
    )

    plot_probs(diff, args, png_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', nargs='+', help='Space-separated list of files to visualize')
    parser.add_argument('-cmap', type=str, default='Reds', help='Matplotlib colormap to use. Some options: (Reds, Blues, RdBu, rainbow, cividis)')
    parser.add_argument('-diff', action='store_true', help='Whether to subtract the probabilities (must be same shape).')
    args = parser.parse_args()

    if args.diff:
        if len(args.input) != 2:
            raise ValueError("If subtracting pmaps, then exactly 2 filenames must be passed.")
        subtract(args)
    else:
        main(args)