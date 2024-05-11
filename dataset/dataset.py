import numpy as np
import torch
import torchvision.transforms as transforms
import random
from torchvision import datasets, transforms
from .all_datasets import isic2019, ICH
import os
import copy
from .dataset_medmnist import PathMNIST,OCTMNIST
from utils.sampling import iid_sampling, non_iid_dirichlet_sampling


def add_noise(args, y_train, dict_users):
	np.random.seed(args.seed)

	gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
	gamma_c_initial = np.random.rand(args.num_users)
	gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
	gamma_c = gamma_s * gamma_c_initial

	y_train_noisy = copy.deepcopy(y_train)

	real_noise_level = np.zeros(args.num_users)
	for i in np.where(gamma_c > 0)[0]:
		sample_idx = np.array(list(dict_users[i]))
		prob = np.random.rand(len(sample_idx))
		noisy_idx = np.where(prob <= gamma_c[i])[0]
		y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, args.num_classes, len(noisy_idx))
		noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
		print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
			i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
		real_noise_level[i] = noise_ratio
	return (y_train_noisy, gamma_s, real_noise_level)

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cuda.matmul.allow_tf32 = False
	torch.backends.cudnn.allow_tf32 = False

def get_dataset(args):
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if args.dataset == 'cifar10':
		data_path = '../data/cifar10'
		args.num_classes = 10
		args.model = 'resnet18'
		trans_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])],
		)
		trans_val = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])],
		)
		dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_train)
		dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)
		n_train = len(dataset_train)
		y_train = np.array(dataset_train.targets)

	elif args.dataset == 'pathmnist':
		data_path = '../data/'
		args.num_classes = 9
		args.model = 'resnet18'
		trans_train = transforms.Compose([
			transforms.RandomCrop(128, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])],
		)
		trans_val = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])],
		)
		dataset_train = PathMNIST(root=data_path, split='train', download=True, size=128, transform=trans_train, mmap_mode='r', as_rgb=True)
		dataset_test = PathMNIST(root=data_path, split='test', download=True, size=128, transform=trans_val, mmap_mode='r', as_rgb=True)
		n_train = len(dataset_train)
		y_train = np.array(dataset_train.targets)
		y_train = y_train.reshape(y_train.shape[0], )
		dataset_train.targets = copy.deepcopy(y_train)

		y_test = np.array(dataset_test.targets)
		y_test = y_test.reshape(y_test.shape[0], )
		dataset_test.targets = copy.deepcopy(y_test)

		
	elif args.dataset == 'cifar100':
		data_path = '../data/cifar100'
		args.num_classes = 100
		args.model = 'resnet34'
		trans_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.507, 0.487, 0.441],
								 std=[0.267, 0.256, 0.276])],
		)
		trans_val = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.507, 0.487, 0.441],
								 std=[0.267, 0.256, 0.276])],
		)
		dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=trans_train)
		dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)
		n_train = len(dataset_train)
		y_train = np.array(dataset_train.targets)

	elif args.dataset == 'clothing1m':
		data_path = os.path.abspath('..') + '/data/clothing1M/'
		args.num_classes = 14
		args.model = 'resnet50'
		trans_train = transforms.Compose([
					transforms.Resize((256, 256)),
					transforms.RandomCrop(224),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				 ])
		trans_val = transforms.Compose([
					transforms.Resize((224, 224)),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				 ])
		dataset_train = Clothing(data_path, trans_train, "train")
		dataset_test = Clothing(data_path, trans_val, "test")
		n_train = len(dataset_train)
		y_train = np.array(dataset_train.targets)

	else:
		exit('Error: unrecognized dataset')

	if args.iid:
		dict_users = iid_sampling(n_train, args.num_users, args.seed)
	else:
		dict_users = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha)

	weights = []
	for i in range(args.num_users):
		weights.append(len(dict_users[i])/len(dataset_train))
	return dataset_train, dataset_test, dict_users



