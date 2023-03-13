import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt


class CIFAR10SimPair(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10SimPair, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.train:
            if self.transform is not None:
                img_1 = self.transform(img)
                img_2 = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img_1, img_2, target
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.data)