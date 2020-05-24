import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np

IMG_EXTENSIONS = ('.npy',)

def make_dataset(path):
    if not os.path.exists(path):
        raise FileExistsError('some subfolders from data set do not exists!')
    samples = []
    for sample in os.listdir(path):
        image  = os.path.join(path, sample)
        samples.append(image)
    return samples

def npy_loader(path):
    samples = np.load(path, allow_pickle=True).item()
    samples['vision'] = torch.from_numpy(samples['vision'])
    samples['behavior'] = torch.from_numpy(samples['behavior']).float()
    return samples


class CY101Dataset(Dataset):
    def __init__(self, root, image_transform=None, loader=npy_loader, device='cpu'):
        if not os.path.exists(root):
            raise FileExistsError('{0} does not exists!'.format(root))

        self.image_transform = lambda vision: torch.cat([image_transform(single_image).unsqueeze(0) for single_image in vision.unbind(0)], dim=0)

        self.samples = make_dataset(root)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.loader = loader
        self.device = device

    def __getitem__(self, index):
        modalities = self.loader(self.samples[index])
        vision = modalities['vision']
        behavior = modalities['behavior']

        if self.image_transform is not None:
            vision = self.image_transform(vision)
            # vision = torch.cat([self.image_transform(single_image).unsqueeze(0) for single_image in vision.unbind(0)], dim=0)

        return vision.to(self.device), behavior.to(self.device)

    def __len__(self):
        return len(self.samples)


def build_dataloader_CY101(opt):
    def crop(im):
        height, width = im.shape[1:]
        width = max(height, width)
        im = im[:, :width, :width]
        return im

    image_transform = transforms.Compose([
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize((opt.height, opt.width)),
        transforms.ToTensor()
    ])


    train_ds = CY101Dataset(
        root=os.path.join(opt.data_dir+'/train'),
        image_transform=image_transform,
        loader=npy_loader,
        device=opt.device
    )

    valid_ds = CY101Dataset(
        root=os.path.join(opt.data_dir+'/test'),
        image_transform=image_transform,
        loader=npy_loader,
        device=opt.device
    )

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    return train_dl, valid_dl

if __name__ == '__main__':
    from options import Options
    opt = Options().parse()
    opt.data_dir = opt.data_dir
    # opt.data_dir ='/Users/ramtin/PycharmProjects/data/CY101NPY'
    tr, va = build_dataloader_CY101(opt)

    import cv2
    for a, b, c, d, e in tr:
        imgs = c[0].unbind(0)
        imgs = list(map(lambda x:(x.permute([1, 2, 0]).cpu().numpy()*255).squeeze().astype(np.uint8), imgs))
        for img in imgs:
            cv2.imshow('l', img)
            cv2.waitKey(0)

