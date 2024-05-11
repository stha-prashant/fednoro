from dataset.dataset_medmnist import PathMNIST
from torchvision import datasets, transforms
data_path = '../data/'
trans_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441],
                            std=[0.267, 0.256, 0.276])],
)
trans_val = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441],
                            std=[0.267, 0.256, 0.276])],
)
dataset_train = PathMNIST(split='train', size=224, download=True, transform=trans_train, root=data_path, as_rgb=True)
dataset_test = PathMNIST(split='test', size=224, download=True, transform=trans_val, root=data_path, as_rgb=True)