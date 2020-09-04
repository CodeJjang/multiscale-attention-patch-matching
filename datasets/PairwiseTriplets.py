from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A

from utils import NormalizeImages
from skimage.transform import resize


class PairwiseTriplets(Dataset):

    def __init__(self, Data, Labels, batch_size, Augmentation, Mode, NegativeMode='Random'):
        self.PositiveIdx = np.squeeze(np.asarray(np.where(Labels == 1)));
        self.NegativeIdx = np.squeeze(np.asarray(np.where(Labels == 0)));

        self.PositiveIdxNo = len(self.PositiveIdx)
        self.NegativeIdxNo = len(self.NegativeIdx)

        self.Data = Data
        self.Labels = Labels

        self.batch_size = batch_size
        self.Augmentation = Augmentation

        self.Mode = Mode
        self.NegativeMode = NegativeMode

        self.ChannelMean1 = Data[:, :, :, 0].mean()
        self.ChannelMean2 = Data[:, :, :, 1].mean()

        self.RowsNo = Data.shape[1]
        self.ColsNo = Data.shape[2]

        self.transform = A.ReplayCompose([
            # A.Transpose(always_apply=False, p=0.5),
            # A.Flip(always_apply=False, p=0.5),
            # A.RandomResizedCrop(self.RowsNo, self.ColsNo,scale=(0.9, 1.1) ,ratio=(0.9, 1.1), interpolation=cv2.INTER_CUBIC,always_apply=False,p=0.5),
            A.Rotate(limit=5, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, always_apply=False,
                     p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True,always_apply=False, p=0.5),
            # A.RandomGamma(gamma_limit=136, always_apply=False, p=0.5),
            # A.JpegCompression(quality_lower=40, quality_upper=100, p=0.5),
            # A.HueSaturationValue(hue_shift_limit=172, sat_shift_limit=20, val_shift_limit=27, always_apply=False, p=0.5)
        ])

    def __len__(self):
        return self.Data.shape[0]

    def __getitem__(self, index):

        # Select pos2 pairs
        PosIdx = np.random.randint(self.PositiveIdxNo, size=self.batch_size)
        PosIdx = self.PositiveIdx[PosIdx]
        PosImages = self.Data[PosIdx, :, :, :].astype(np.float32)

        # imshow(torchvision.utils.make_grid(PosImages[0,:,:,0]))
        # plt.imshow(np.squeeze(PosImages[2040, :, :, :]));  # plt.show()

        pos1 = PosImages[:, :, :, 0]
        pos2 = PosImages[:, :, :, 1]

        for i in range(0, PosImages.shape[0]):

            # Flip LR
            if (np.random.uniform(0, 1) > 0.5) & self.Augmentation["HorizontalFlip"]:
                pos1[i,] = np.fliplr(pos1[i,])
                pos2[i,] = np.fliplr(pos2[i,])

            # flip UD
            if (np.random.uniform(0, 1) > 0.5) & self.Augmentation["VerticalFlip"]:
                pos1[i,] = np.flipud(pos1[i,])
                pos2[i,] = np.flipud(pos2[i,])

            # test
            if (np.random.uniform(0, 1) > 0) & self.Augmentation["Test"]['Do']:
                # plt.imshow(pos1[i,:,:],cmap='gray');plt.show();
                data = self.transform(image=pos1[i, :, :])
                pos1[i,] = data['image']
                pos2[i,] = A.ReplayCompose.replay(data['replay'], image=pos2[i, :, :])['image']

            # rotate:0, 90, 180,270,
            if self.Augmentation["Rotate90"]:
                idx = np.random.randint(low=0, high=4, size=1)[0]  # choose rotation
                pos1[i,] = np.rot90(pos1[i,], idx)
                pos2[i,] = np.rot90(pos2[i,], idx)

            # random crop
            if (np.random.uniform(0, 1) > 0.5) & self.Augmentation["RandomCrop"]['Do']:
                dx = np.random.uniform(self.Augmentation["RandomCrop"]['MinDx'],
                                       self.Augmentation["RandomCrop"]['MaxDx'])
                dy = np.random.uniform(self.Augmentation["RandomCrop"]['MinDy'],
                                       self.Augmentation["RandomCrop"]['MaxDy'])

                dx = dy

                x0 = int(dx * self.ColsNo)
                y0 = int(dy * self.RowsNo)

                # ShowRowImages(pos1[0:1,:,:])
                # plt.imshow(pos1[i,:,:],cmap='gray');plt.show();
                # aa = pos1[i,y0:,x0:]

                pos1[i,] = resize(pos1[i, y0:, x0:], (self.RowsNo, self.ColsNo))

                # ShowRowImages(pos1[0:1, :, :])

                pos2[i,] = resize(pos2[i, y0:, x0:], (self.RowsNo, self.ColsNo))

        Result = dict()

        if (self.Mode == 'Pairwise') | (self.Mode == 'PairwiseRot'):
            pos1 -= self.ChannelMean1
            pos2 -= self.ChannelMean2

            Result['pos1'] = NormalizeImages(pos1)
            Result['pos2'] = NormalizeImages(pos2)

            if self.Mode == 'PairwiseRot':
                Rot1 = np.random.randint(low=0, high=4, size=pos1.shape[0])  # choose rotation
                Rot2 = np.random.randint(low=0, high=4, size=pos1.shape[0])  # choose rotation

                Result['RotPos1'] = np.zeros(pos1.shape, Result['pos1'].dtype)
                Result['RotPos2'] = np.zeros(pos2.shape, Result['pos1'].dtype)

                # ShowRowImages(pos1[0:1,:,:])
                # plt.imshow(pos1[i,:,:],cmap='gray');plt.show();plt.show()
                for k in range(0, pos1.shape[0]):
                    Result['RotPos1'][k,] = np.rot90(Result['pos1'][k,], Rot1[k])
                    Result['RotPos2'][k,] = np.rot90(Result['pos2'][k,], Rot2[k])

                Result['RotLabel1'] = Rot1.astype(np.int64)
                Result['RotLabel2'] = Rot2.astype(np.int64)

            return Result

        if self.Mode == 'Triplet':
            Result['pos1'] = NormalizeImages(pos1)
            Result['pos2'] = NormalizeImages(pos2)
            Result['neg1'] = NormalizeImages(neg1)
            Result['neg2'] = NormalizeImages(neg2)

            return Result
