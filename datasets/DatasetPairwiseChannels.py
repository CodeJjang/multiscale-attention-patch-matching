import albumentations as A
import cv2
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset

from util.utils import normalize_image


class DatasetPairwiseTriplets(Dataset):

    def __init__(self, data, labels, batch_size, augmentations, mode, negative_mining_mode='Random'):
        self.pos_indices = np.squeeze(np.asarray(np.where(labels == 1)))
        self.neg_indices = np.squeeze(np.asarray(np.where(labels == 0)))

        self.pos_amount = len(self.pos_indices)
        self.neg_amount = len(self.neg_indices)

        self.data = data
        self.labels = labels

        self.batch_size = batch_size
        self.augmentations = augmentations

        self.mode = mode
        self.negative_mining_mode = negative_mining_mode

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
        # Select pos pairs
        pos_idx = np.random.randint(self.pos_amount, size=int(self.batch_size))
        # pos_idx = np.random.randint(self.pos_amount, size=int(self.batch_size / 2))

        pos_idx = self.pos_indices[pos_idx]
        pos_images = self.data[pos_idx, :, :, :].astype(np.float32)

        pos1 = pos_images[:, :, :, 0]
        pos2 = pos_images[:, :, :, 1]

        pos1 = np.concatenate([pos1, pos2], axis=0)
        pos2 = pos1.copy()

        for i in range(0, pos1.shape[0]):

            # flip LR
            if (np.random.uniform(0, 1) > 0.5) and self.augmentations.get("HorizontalFlip"):
            # pos1[i,] = np.fliplr(pos1[i,])
                pos2[i,] = np.fliplr(pos2[i,])

            # flip UD
            if (np.random.uniform(0, 1) > 0.5) and self.augmentations.get("VerticalFlip"):
                # pos1[i,] = np.flipud(pos1[i,])
                pos2[i,] = np.flipud(pos2[i,])

            # test augmentations
            if self.augmentations.get("Test"):
                data = self.transform(image=pos1[i, :, :])
                pos1[i,] = data['image']
                pos2[i,] = A.ReplayCompose.replay(data['replay'], image=pos2[i, :, :])['image']

            # rotate
            if self.augmentations.get("Rotate90"):
                idx = np.random.randint(low=0, high=4, size=1)[0]  # choose rotation
                # pos1[i,] = np.rot90(pos1[i,], idx)
                pos2[i,] = np.rot90(pos2[i,], idx)

            # random crop
            if (np.random.uniform(0, 1) > 0.5) & self.augmentations.get("RandomCrop", {}).get('Do'):
                dx = np.random.uniform(self.augmentations.get("RandomCrop", {}).get('MinDx'),
                                       self.augmentations.get("RandomCrop", {}).get('MaxDx'))
                dy = np.random.uniform(self.augmentations.get("RandomCrop", {}).get('MinDy'),
                                       self.augmentations.get("RandomCrop", {}).get('MaxDy'))

                dx = dy

                x0 = int(dx * self.data_width)
                y0 = int(dy * self.data_height)

                # pos1[i,] = resize(pos1[i, y0:, x0:], (self.data_height, self.data_width))

                pos2[i,] = resize(pos2[i, y0:, x0:], (self.data_height, self.data_width))

        res = dict()

        pos1 -= self.channel_mean1
        pos2 -= self.channel_mean2

        res['pos1'] = normalize_image(pos1)
        res['pos2'] = normalize_image(pos2)

        return res
