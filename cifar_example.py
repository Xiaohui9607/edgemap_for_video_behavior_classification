import torch
import torchvision
import torchvision.transforms as transforms
from PDBF import graypdbfs

transform = transforms.Compose(
    [
        transforms.Lambda(lambda x:np.concatenate([x, graypdbfs(x, [1,3,5])*255], axis=2)),
        transforms.ToTensor(),
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img,name):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(name+'.png', bbox_inches='tight')


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
rgb, mix, edge1, edge2, edge3 = images[:,:3, ...], images[:, 3:6,...], images[:, 3:4,...], images[:, 4:5,...], images[:, 5:6,...]

# show images
imshow(torchvision.utils.make_grid(rgb), 'rgb')
imshow(torchvision.utils.make_grid(mix),'mix')
imshow(torchvision.utils.make_grid(edge1),'1')
imshow(torchvision.utils.make_grid(edge2),'3')
imshow(torchvision.utils.make_grid(edge3),'5')
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))