import ntpath
import os
import pathlib
import warnings

import GPUtil
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from networks.PatchMatchingNetwork import PatchMatchingNetwork
from util.utils import load_model, normalize_image, evaluate_network

warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")


def display(rgb, attn1, nir_orig, attn2):
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(rgb)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Input Image')

    ax[0, 1].imshow(rgb)
    ax[0, 1].imshow(attn1, alpha=0.25, cmap='jet')
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Attention')

    ax[1, 0].imshow(nir_orig, cmap="gray")
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Input Image')

    ax[1, 1].imshow(nir_orig, cmap="gray")
    ax[1, 1].imshow(attn2, alpha=0.25, cmap='jet')
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Attention')

    plt.show()


def get_file_name(path):
    head, tail = ntpath.split(path)
    fname = tail or ntpath.basename(head)
    return fname.split('.')[0]


def save_image(img, fname, out_folder):
    fname += '.png'
    mpimg.imsave(os.path.join(out_folder, fname), img)


def generate_attn_heatmaps(net, imgs, outpath, device, disp=True):
    net.eval()
    rgb_attentions = []
    nir_attentions = []
    for rgb_path, nir_path in imgs:
        print('Working on:', rgb_path)

        dir = rgb_path.split('\\')[-2]
        curr_outpath = os.path.join(outpath, dir)

        rgb = mpimg.imread(rgb_path)
        rgb_gray_orig = cv2.imread(rgb_path)
        rgb_gray_orig = cv2.cvtColor(rgb_gray_orig, cv2.COLOR_BGR2GRAY)
        nir_orig = mpimg.imread(nir_path)
        pathlib.Path(curr_outpath).mkdir(parents=True, exist_ok=True)

        rgb_gray = rgb_gray_orig.copy().reshape(1, 1, rgb_gray_orig.shape[0], rgb_gray_orig.shape[1])
        nir = nir_orig.copy().reshape(1, 1, nir_orig.shape[0], nir_orig.shape[1])
        rgb_gray = torch.from_numpy(normalize_image(rgb_gray.astype(np.float32)))
        nir = torch.from_numpy(normalize_image(nir.astype(np.float32)))

        emb = evaluate_network(net, rgb_gray, nir, device, 800)
        emb1_attn = np.array(emb['Emb1Attention']).squeeze()
        emb2_attn = np.array(emb['Emb2Attention']).squeeze()

        _, emb1_attn = emb1_attn[0], emb1_attn[1:]

        emb1_attn = np.mean(emb1_attn.reshape(8 * 8, 128), axis=1)
        indices = emb1_attn.argsort()[:int(-0.9 * emb1_attn.shape[0])]
        emb1_attn[indices] = 0
        emb1_attn = emb1_attn.reshape(8, 8)
        emb1_attn = 255 * (emb1_attn - emb1_attn.min()) / (emb1_attn.max() - emb1_attn.min())
        emb1_attn = np.uint8(emb1_attn)
        emb1_attn = cv2.resize(emb1_attn, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)

        _, emb2_attn = emb2_attn[0], emb2_attn[1:]

        emb2_attn = np.mean(emb2_attn.reshape(8 * 8, 128), axis=1)
        indices = emb2_attn.argsort()[:int(-0.9 * emb2_attn.shape[0])]
        emb2_attn[indices] = 0
        emb2_attn = emb2_attn.reshape(8, 8)
        emb2_attn = 255 * (emb2_attn - emb2_attn.min()) / (emb2_attn.max() - emb2_attn.min())
        emb2_attn = np.uint8(emb2_attn)
        emb2_attn = cv2.resize(emb2_attn, (nir_orig.shape[1], nir_orig.shape[0]), interpolation=cv2.INTER_CUBIC)

        rgb_attentions.append(emb1_attn.copy())
        nir_attentions.append(emb2_attn.copy())
        if disp:
            display(rgb, emb1_attn, nir_orig, emb2_attn)
            plt.close()

    max_h = max([emb.shape[0] for emb in rgb_attentions])
    max_w = max([emb.shape[1] for emb in rgb_attentions])
    padded_rgb = np.zeros((len(rgb_attentions), max_h, max_w))
    for i, emb in enumerate(rgb_attentions):
        padded_rgb[i, :emb.shape[0], :emb.shape[1]] = emb
    padded_nir = np.zeros((len(nir_attentions), max_h, max_w))
    for i, emb in enumerate(nir_attentions):
        padded_nir[i, :emb.shape[0], :emb.shape[1]] = emb
    rgb_attentions = np.array(padded_rgb).max(axis=0)
    nir_attentions = np.array(padded_nir).max(axis=0)

    dpi = 80
    figsize = rgb.shape[1] / dpi, rgb.shape[0] / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb_attentions, alpha=1, cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join(outpath, 'avg_rgb' + '_attention' + '.png'))
    plt.close()
    figsize = nir_orig.shape[1] / dpi, nir_orig.shape[0] / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(nir_attentions, alpha=1, cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join(outpath, 'avg_nir' + '_attention' + '.png'))
    plt.close()


def read_data_fnames(data_path):
    fnames = []
    for _, subdirs, _ in os.walk(data_path):
        for subdir in subdirs:
            subdir_fpath = os.path.join(data_path, subdir)
            for _, _, files in os.walk(subdir_fpath):
                for f in files:
                    if not '_rgb' in f:
                        continue
                    fpath_rgb = os.path.join(subdir_fpath, f)
                    fpath_nir = fpath_rgb.replace('_rgb', '_nir')
                    fnames.append((fpath_rgb, fpath_nir))

    return fnames


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # "cuda:0"
    num_gpus = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

    print(device)

    models_dir_name = '../artifacts/symmetric_enc_transformer_visnir_10/models/'
    best_fname = 'best_model'

    output_attention_weights = True
    net = PatchMatchingNetwork(output_attention_weights)

    net, optimizer, LowestError, StartEpoch, scheduler, LodedNegativeMiningMode = load_model(net, True,
                                                                                             models_dir_name,
                                                                                             best_fname,
                                                                                             True, device)
    if num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    outpath = "D:\\multisensor\\attentions\\"

    imgs = read_data_fnames("D:\\multisensor\\datasets\\Vis-Nir\\data")
    generate_attn_heatmaps(net, imgs, outpath, device, disp=False)


if __name__ == '__main__':
    main()
