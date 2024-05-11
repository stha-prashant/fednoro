import os
import copy
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from collections import Counter

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.options import args_parser
from utils.local_training import LocalUpdate, globaltest
from utils.FedAvg import FedAvg, DaAgg
from utils.utils import set_seed, set_output_files, get_output, get_current_consistency_weight
from dataset.dataset import add_noise

import neptune
from dataset.dataset import get_dataset
from model.build_model import build_model
np.set_printoptions(threshold=np.inf)


"""
Major framework of noise FL
"""


if __name__ == '__main__':
	args = args_parser()
	args.num_users = args.n_clients
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	args.device = "cuda" if torch.cuda.is_available() else "cpu"

	# ------------------------------ deterministic or not ------------------------------
	if args.deterministic:
		cudnn.benchmark = False
		cudnn.deterministic = True
		set_seed(args.seed)

	# ------------------------------ output files ------------------------------
	writer, models_dir, exp_dir = set_output_files(args)

	# ------------------------------ dataset ------------------------------
	dataset_train, dataset_test, dict_users = get_dataset(args)
	logging.info(
		f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
	logging.info(
		f"test: {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")

	# --------------------- Add Noise ---------------------------
	y_train = np.array(dataset_train.targets)
	y_train_noisy, gamma_s, real_noise_level = add_noise(
		args, y_train, dict_users)
	dataset_train.targets = y_train_noisy

	# --------------------- Build Models ---------------------------
	netglob = build_model(args)
	user_id = list(range(args.n_clients))
	trainer_locals = []
	for id in user_id:
		trainer_locals.append(LocalUpdate(
			args, id, dataset_train, dict_users[id]))

	# ------------------------------ begin training ------------------------------
	set_seed(args.seed)
	logging.info("\n ---------------------begin training---------------------")
	best_performance = 0.

	if args.wandb:
		distribution = f'alpha_{args.alpha}' if not args.iid else 'iid'
		wandb_name = f'fednoro_s1[{args.s1}]_E[{args.rounds}]_noisy[{args.level_n_system}]_level[{args.level_n_lowerb}]'
		run = neptune.init_run(
			project="fednl/fednl",
			api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDA5OTRhYi0zYjkzLTQ4NTEtOTAwYS1hNzZhYjQ2ZjBlN2UifQ==",
			name = f"{wandb_name}_{args.dataset}_{args.n_clients}_{args.frac}_{distribution}_{args.seed}",
		)  
		run['parameters'] = args
		run["utils"].upload_files("utils/*.py")
		run["dataset"].upload_files("dataset/*.py")
		run["model"].upload_files("dataset/*.py")
	# ------------------------ Stage 1: warm up ------------------------ 
	if args.warm:
		for rnd in range(args.s1):
			w_locals, loss_locals = [], []
			m = max(int(args.frac * args.num_users), 1)
			print("Random Selection")
			user_id = np.random.choice(range(args.num_users), m, replace=False)
			for idx in user_id:  # training over the subset
				local = trainer_locals[idx]
				w_local, loss_local = local.train_LA(
					net=copy.deepcopy(netglob).to(args.device), writer=writer)

				# store every updated model
				w_locals.append(copy.deepcopy(w_local))
				loss_locals.append(copy.deepcopy(loss_local))

			w_locals_last = copy.deepcopy(w_locals)
			dict_len = [len(dict_users[idx]) for idx in user_id]
			w_glob_fl = FedAvg(w_locals, dict_len)
			netglob.load_state_dict(copy.deepcopy(w_glob_fl))

			pred = globaltest(copy.deepcopy(netglob).to(
				args.device), dataset_test, args)
			acc = accuracy_score(dataset_test.targets, pred)
			bacc = balanced_accuracy_score(dataset_test.targets, pred)
			cm = confusion_matrix(dataset_test.targets, pred)
			logging.info(
				"******** round: %d, acc: %.4f, bacc: %.4f ********" % (rnd, acc, bacc))
			logging.info(cm)
			writer.add_scalar(f'test/acc', acc, rnd)
			writer.add_scalar(f'test/bacc', bacc, rnd)

			# save model
			if bacc > best_performance:
				best_performance = bacc
			logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')
			logging.info('\n')
		torch.save(netglob.state_dict(),  models_dir +
				   f'/stage1_model_{rnd}.pth')

	#  ------------------------ client selection ------------------------
	model_path = os.path.join(models_dir, f"stage1_model_{args.s1-1}.pth")
	logging.info(
		f"********************** load model from: {model_path} **********************")
	netglob.load_state_dict(torch.load(model_path))
	loader = DataLoader(dataset=dataset_train, batch_size=32,
						shuffle=False, num_workers=4)
	criterion = nn.CrossEntropyLoss(reduction='none')
	local_output, loss = get_output(
		loader, netglob.to(args.device), args, False, criterion)
	metrics = np.zeros((args.n_clients, args.num_classes)).astype("float")
	num = np.zeros((args.n_clients, args.num_classes)).astype("float")
	for id in range(args.n_clients):
		idxs = dict_users[id]
		for idx in idxs:
			c = dataset_train.targets[idx]
			num[id, c] += 1
			metrics[id, c] += loss[idx]
	metrics = metrics / num
	for i in range(metrics.shape[0]):
		for j in range(metrics.shape[1]):
			if np.isnan(metrics[i, j]):
				metrics[i, j] = np.nanmin(metrics[:, j])
	for j in range(metrics.shape[1]):
		metrics[:, j] = (metrics[:, j]-metrics[:, j].min()) / \
			(metrics[:, j].max()-metrics[:, j].min())
	logging.info("metrics:")
	logging.info(metrics)

	vote = []
	for i in range(9):
		gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
		gmm_pred = gmm.predict(metrics)
		noisy_clients = np.where(gmm_pred == np.argmax(gmm.means_.sum(1)))[0]
		noisy_clients = set(list(noisy_clients))
		vote.append(noisy_clients)
	cnt = []
	for i in vote:
		cnt.append(vote.count(i))
	noisy_clients = list(vote[cnt.index(max(cnt))])

	logging.info(
		f"selected noisy clients: {noisy_clients}, real noisy clients: {np.where(gamma_s>0.)[0]}")
	clean_clients = list(set(range(args.num_users)) - set(noisy_clients))
	logging.info(f"selected clean clients: {clean_clients}")

	# ------------------------ Stage 2: ------------------------ 
	BACC = []
	accuracy_scores = []
	for rnd in range(args.s1, args.rounds):
		w_locals, loss_locals = [], []
		weight_kd = get_current_consistency_weight(
			rnd, args.begin, args.end) * args.a
		writer.add_scalar(f'train/w_kd', weight_kd, rnd)
		m = max(int(args.frac * args.num_users), 1)
		print("Random Selection")
		user_id = np.random.choice(range(args.num_users), m, replace=False)
		for idx in user_id:  # training over the subset
			local = trainer_locals[idx]
			if idx in clean_clients:
				w_local, loss_local = local.train_LA(
					net=copy.deepcopy(netglob).to(args.device), writer=writer)
			elif idx in noisy_clients:
				w_local, loss_local = local.train_FedNoRo(
					student_net=copy.deepcopy(netglob).to(args.device), teacher_net=copy.deepcopy(netglob).to(args.device), writer=writer, weight_kd=weight_kd)
			# store every updated model
			w_locals.append(copy.deepcopy(w_local))
			loss_locals.append(copy.deepcopy(loss_local))
			assert len(w_locals) == len(loss_locals)

		dict_len = [len(dict_users[idx]) for idx in user_id]
		selected_noisy_client = [idx for idx in user_id if idx in noisy_clients]
		selected_clean_client = [idx for idx in user_id if idx in clean_clients]
		w_glob_fl = DaAgg(
			w_locals, dict_len, selected_clean_client, selected_noisy_client, user_id)
		netglob.load_state_dict(copy.deepcopy(w_glob_fl))

		pred = globaltest(copy.deepcopy(netglob).to(
			args.device), dataset_test, args)
		acc = accuracy_score(dataset_test.targets, pred)
		bacc = balanced_accuracy_score(dataset_test.targets, pred)
		cm = confusion_matrix(dataset_test.targets, pred)
		logging.info(
			"******** round: %d, acc: %.4f, bacc: %.4f ********" % (rnd, acc, bacc))
		logging.info(cm)
		writer.add_scalar(f'test/acc', acc, rnd)
		writer.add_scalar(f'test/bacc', bacc, rnd)
		BACC.append(bacc)
		accuracy_scores.append(acc)

		if args.wandb:
			run['test/accuracy'].append(acc)
			run['test/max_accuracy'].append(np.max(accuracy_scores))

		# save model
		if bacc > best_performance:
			best_performance = bacc
		logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')
		logging.info('\n')
	torch.save(netglob.state_dict(),  models_dir+f'/stage2_model_{rnd}.pth')

	BACC = np.array(BACC)
	logging.info("last:")
	logging.info(BACC[-10:].mean())
	logging.info("best:")
	logging.info(BACC.max())

	torch.cuda.empty_cache()