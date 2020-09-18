from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A

from utils import NormalizeImages
from skimage.transform import resize


class PairwiseTriplets(Dataset):

    def __init__(self, data, labels, batch_size, augmentations, mode, negative_mode='Random'):
        self.positive_indices = np.squeeze(np.asarray(np.where(labels == 1)));
        self.negative_indices = np.squeeze(np.asarray(np.where(labels == 0)));

        self.data = data
        self.labels = labels

        self.batch_size = batch_size
        self.augmentations = augmentations

        self.mode = mode
        self.negative_mode = negative_mode

        self.data_height = data.shape[1]
        self.data_width = data.shape[2]

        self.transform = A.ReplayCompose([
            # A.Transpose(always_apply=False, p=0.5),
            # A.Flip(always_apply=False, p=0.5),
            # A.RandomResizedCrop(self.data_height, self.data_width,scale=(0.9, 1.1) ,ratio=(0.9, 1.1), interpolation=cv2.INTER_CUBIC,always_apply=False,p=0.5),
            A.Rotate(limit=5, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, always_apply=False,
                     p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True,always_apply=False, p=0.5),
            # A.RandomGamma(gamma_limit=136, always_apply=False, p=0.5),
            # A.JpegCompression(quality_lower=40, quality_upper=100, p=0.5),
            # A.HueSaturationValue(hue_shift_limit=172, sat_shift_limit=20, val_shift_limit=27, always_apply=False, p=0.5)
        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        # Select pos2 pairs
        positive_idx = np.random.randint(len(self.positive_indices), size=self.batch_size)
        positive_idx = self.positive_indices[positive_idx]
        positive_images = self.data[positive_idx, :, :, :].astype(np.float32)

        # imshow(torchvision.utils.make_grid(positive_images[0,:,:,0]))
        # plt.imshow(np.squeeze(positive_images[2040, :, :, :]));  # plt.show()

        pos1 = positive_images[:, :, :, 0]
        pos2 = positive_images[:, :, :, 1]

        for i in range(0, positive_images.shape[0]):

            # Flip LR
            if (np.random.uniform(0, 1) > 0.5) & self.augmentations["HorizontalFlip"]:
                pos1[i,] = np.fliplr(pos1[i,])
                pos2[i,] = np.fliplr(pos2[i,])

            # flip UD
            if (np.random.uniform(0, 1) > 0.5) & self.augmentations["VerticalFlip"]:
                pos1[i,] = np.flipud(pos1[i,])
                pos2[i,] = np.flipud(pos2[i,])

            # test
            if (np.random.uniform(0, 1) > 0) & self.augmentations["Test"]['Do']:
                # plt.imshow(pos1[i,:,:],cmap='gray');plt.show();
                data = self.transform(image=pos1[i, :, :])
                pos1[i,] = data['image']
                pos2[i,] = A.ReplayCompose.replay(data['replay'], image=pos2[i, :, :])['image']

            # rotate:0, 90, 180,270,
            if self.augmentations["Rotate90"]:
                idx = np.random.randint(low=0, high=4, size=1)[0]  # choose rotation
                pos1[i,] = np.rot90(pos1[i,], idx)
                pos2[i,] = np.rot90(pos2[i,], idx)

            # random crop
            if (np.random.uniform(0, 1) > 0.5) & self.augmentations["RandomCrop"]['Do']:
                dx = np.random.uniform(self.augmentations["RandomCrop"]['MinDx'],
                                       self.augmentations["RandomCrop"]['MaxDx'])
                dy = np.random.uniform(self.augmentations["RandomCrop"]['MinDy'],
                                       self.augmentations["RandomCrop"]['MaxDy'])

                dx = dy

                x0 = int(dx * self.data_width)
                y0 = int(dy * self.data_height)

                # ShowRowImages(pos1[0:1,:,:])
                # plt.imshow(pos1[i,:,:],cmap='gray');plt.show();
                # aa = pos1[i,y0:,x0:]

                pos1[i,] = resize(pos1[i, y0:, x0:], (self.data_height, self.data_width))

                # ShowRowImages(pos1[0:1, :, :])

                pos2[i,] = resize(pos2[i, y0:, x0:], (self.data_height, self.data_width))

        result = dict()

        if (self.mode == 'Pairwise') | (self.mode == 'PairwiseRot'):
            pos1 -= data[:, :, :, 0].mean()
            pos2 -= data[:, :, :, 1].mean()

            result['pos1'] = NormalizeImages(pos1)
            result['pos2'] = NormalizeImages(pos2)

            if self.mode == 'PairwiseRot':
                Rot1 = np.random.randint(low=0, high=4, size=pos1.shape[0])  # choose rotation
                Rot2 = np.random.randint(low=0, high=4, size=pos1.shape[0])  # choose rotation

                result['RotPos1'] = np.zeros(pos1.shape, result['pos1'].dtype)
                result['RotPos2'] = np.zeros(pos2.shape, result['pos1'].dtype)

                # ShowRowImages(pos1[0:1,:,:])
                # plt.imshow(pos1[i,:,:],cmap='gray');plt.show();plt.show()
                for k in range(0, pos1.shape[0]):
                    result['RotPos1'][k,] = np.rot90(result['pos1'][k,], Rot1[k])
                    result['RotPos2'][k,] = np.rot90(result['pos2'][k,], Rot2[k])

                result['RotLabel1'] = Rot1.astype(np.int64)
                result['RotLabel2'] = Rot2.astype(np.int64)

            return result

        if self.mode == 'Triplet':
            result['pos1'] = NormalizeImages(pos1)
            result['pos2'] = NormalizeImages(pos2)
            result['neg1'] = NormalizeImages(neg1)
            result['neg2'] = NormalizeImages(neg2)

            return result
