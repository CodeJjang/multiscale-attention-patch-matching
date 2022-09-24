import albumentations as A
import cv2
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset

from util.utils import normalize_image


class SelfSupervisionPairwiseDataset(Dataset):

    def __init__(self, data, batch_size, augmentations):
        self.data = data

        self.batch_size = batch_size
        self.augmentations = augmentations

        self.channel_mean1 = data[:, :, :, 0].mean()
        self.channel_mean2 = data[:, :, :, 1].mean()

        self.data_height = data.shape[1]
        self.data_width = data.shape[2]

        self.transform = A.ReplayCompose([
            A.Rotate(limit=5, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, always_apply=False,
                     p=0.5),
            A.HorizontalFlip(always_apply=False, p=0.5),
            A.VerticalFlip(always_apply=False, p=0.5),
        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        batch_indices = np.random.randint(len(self.data), size=self.batch_size)

        batch_images = self.data[batch_indices].astype(np.float32)

        im1_indices = np.random.randint(self.batch_size, size=self.batch_size // 2)
        im2_indices = np.random.randint(self.batch_size, size=self.batch_size // 2)

        # Shuffle pairs
        im1 = batch_images[im1_indices, :, :, 0]
        im2 = batch_images[im2_indices, :, :, 1]

        im1 -= self.channel_mean1
        im2 -= self.channel_mean2

        # Mix examples in each batch to have half rgb half nir
        im1_mixed = np.concatenate([im1, im2], axis=0)
        im2_mixed = np.concatenate([im1, im2], axis=0)

        shuffle_indices = np.random.randint(self.batch_size, size=self.batch_size)

        im1_mixed = im1_mixed[shuffle_indices]
        im2_mixed = im2_mixed[shuffle_indices]

        im1, im2 = im1_mixed, im2_mixed

        for i in range(0, batch_images.shape[0]):

            # flip LR
            if (np.random.uniform(0, 1) > 0.5) and self.augmentations.get("HorizontalFlip"):
                im1[i,] = np.fliplr(im1[i,])
                im2[i,] = np.fliplr(im2[i,])

            # flip UD
            if (np.random.uniform(0, 1) > 0.5) and self.augmentations.get("VerticalFlip"):
                im1[i,] = np.flipud(im1[i,])
                im2[i,] = np.flipud(im2[i,])

            # test augmentations
            if self.augmentations.get("Test"):
                data = self.transform(image=im1[i, :, :])
                im1[i,] = data['image']
                im2[i,] = A.ReplayCompose.replay(data['replay'], image=im2[i, :, :])['image']

            # rotate
            if self.augmentations.get("Rotate90"):
                idx = np.random.randint(low=0, high=4, size=1)[0]  # choose rotation
                im1[i,] = np.rot90(im1[i,], idx)
                im2[i,] = np.rot90(im2[i,], idx)

            # random crop
            if (np.random.uniform(0, 1) > 0.5) & self.augmentations.get("RandomCrop", {}).get('Do'):
                dx = np.random.uniform(self.augmentations.get("RandomCrop", {}).get('MinDx'),
                                       self.augmentations.get("RandomCrop", {}).get('MaxDx'))
                dy = np.random.uniform(self.augmentations.get("RandomCrop", {}).get('MinDy'),
                                       self.augmentations.get("RandomCrop", {}).get('MaxDy'))

                dx = dy

                x0 = int(dx * self.data_width)
                y0 = int(dy * self.data_height)

                im1[i,] = resize(im1[i, y0:, x0:], (self.data_height, self.data_width))
                im2[i,] = resize(im2[i, y0:, x0:], (self.data_height, self.data_width))

        res = dict()

        res['pos1'] = normalize_image(im1)
        res['pos2'] = normalize_image(im2)

        return res
