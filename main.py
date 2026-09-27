import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'ATF_UAD' in args.model or 'DC_UAD' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	print("dataset:{0}, train len:{1}, test len:{2}, dim: {3}".format(dataset, loader[0].shape[0], loader[1].shape[0],
																	  loader[0].shape[1]))
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	# DC_UAD模型使用float类型以与Mamba兼容，其他模型使用double类型
	if modelname == 'DC_UAD':
		model = model_class(dims).float()
	else:
		model = model_class(dims).double()
	
	# 将模型移动到GPU（如果可用）
	if torch.cuda.is_available():
		model = model.cuda()
	
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname, weights_only=False)
		
		# 处理模型状态字典的兼容性问题
		model_state_dict = checkpoint['model_state_dict']
		current_model_dict = model.state_dict()
		
		# 过滤掉不匹配的键
		filtered_dict = {}
		for k, v in model_state_dict.items():
			if k in current_model_dict:
				if current_model_dict[k].shape == v.shape:
					filtered_dict[k] = v
				else:
					print(f"Skipping {k} due to shape mismatch: {v.shape} vs {current_model_dict[k].shape}")
			else:
				print(f"Skipping unexpected key: {k}")
		
		# 加载兼容的参数
		model.load_state_dict(filtered_dict, strict=False)
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, scheduler, e, accuracy_list):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            save_model(model, optimizer, scheduler, e, accuracy_list)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save_model(model, optimizer, scheduler, e, accuracy_list)
            self.counter = 0


def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1]
	if 'DAGMM' in model.name:
		l = nn.MSELoss(reduction = 'none')
		compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
		n = epoch + 1; w_size = model.n_window
		l1s = []; l2s = []
		if training:
			for d in data:
				_, x_hat, z, gamma = model(d)
				l1, l2 = l(x_hat, d), l(gamma, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1) + torch.mean(l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s = []
			for d in data: 
				_, x_hat, _, _ = model(d)
				ae1s.append(x_hat)
			ae1s = torch.stack(ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	if 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []; res = []
		if training:
			for d in data:
				ae, ats = model(d)
				# res.append(torch.mean(ats, axis=0).view(-1))
				l1 = l(ae, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			ae1s, y_pred = [], []
			for d in data: 
				ae1 = model(d)
				y_pred.append(ae1[-1])
				ae1s.append(ae1)
			ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
			loss = torch.mean(l(ae1s, data), axis=1)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'OmniAnomaly' in model.name:
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d in data:
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []
		if training:
			for i, d in enumerate(data):
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, h if i else None)
				else:
					x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			xs = []
			with torch.no_grad():
				for d in data:
					if 'MTAD_GAT' in model.name:
						x, h = model(d, None)
					else:
						x = model(d)
					xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'MAD_GAN' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bcel = nn.BCELoss(reduction = 'mean')
		msel = nn.MSELoss(reduction = 'mean')
		real_label, fake_label = torch.FloatTensor([0.9]), torch.FloatTensor([0.1]) # label smoothing
		real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
		n = epoch + 1; w_size = model.n_window
		mses, gls, dls = [], [], []
		if training:
			for d in data:
				# training discriminator
				model.discriminator.zero_grad()
				_, real, fake = model(d)
				dl = bcel(real, real_label) + bcel(fake, fake_label)
				dl.backward()
				model.generator.zero_grad()
				optimizer.step()
				# training generator
				z, _, fake = model(d)
				mse = msel(z, d) 
				gl = bcel(fake, real_label)
				tl = gl + mse
				tl.backward()
				model.discriminator.zero_grad()
				optimizer.step()
				mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
				# tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
			return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
		else:
			outputs = []
			for d in data: 
				z, _, _ = model(d)
				outputs.append(z)
			outputs = torch.stack(outputs)
			y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(outputs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			with torch.no_grad():
				for d, _ in dataloader:
					window = d.permute(1, 0, 2)
					elem = window[-1, :, :].view(1, bs, feats)
					z = model(window, elem)
					if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
			return loss.detach().numpy(), z.detach().numpy()[0]
	elif 'ATF_UAD' in model.name:
		l = nn.MSELoss(reduction='none')
		data_x = torch.DoubleTensor(data);
		dataset = TensorDataset(data_x, data_x)
		
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size=bs)
		n = epoch + 1;
		w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats) # (1, batch_size, dim)
				z, output_list = model(window)
				z = z[-1, :, :].view(1, local_bs, feats)
				l1 = l(z, elem) - model.sigma * l(output_list[0], output_list[1])
				l2 = l(output_list[0], output_list[1])
				l1s.append(torch.mean(l1).item())
				l2s.append(torch.mean(l2).item())
				loss1 = torch.mean(l1)
				loss2 = torch.mean(l2)
				optimizer.zero_grad()
				loss2.backward(retain_graph=True)
				loss1.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			with torch.no_grad():
				for d, _ in dataloader:
					window = d.permute(1, 0, 2)
					elem = window[-1, :, :].view(1, bs, feats)
					z, _ = model(window)
					z = z[-1, :, :].view(1, bs, feats)
			loss = l(z, elem)[0]
			return loss.detach().numpy(), z.detach().numpy()[0]
	elif 'DC_UAD' in model.name:
		loss_func1 = nn.MSELoss(reduction='none')
		# DC_UAD使用float类型数据
		data_x = data.detach().clone().to(torch.float32);
		print(data_x.shape)
		# 将数据移动到与模型相同的设备
		if torch.cuda.is_available():
			data_x = data_x.cuda()
		dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else model.batch
		dataloader = DataLoader(dataset, batch_size=bs)
		n = epoch + 1;
		w_size = model.n_window
		l1s, l2s, l3s = [], [], []
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats) # (1, batch_size, dim)
				z, output_list = model(window)
				
				# 处理输出维度 - 使用z作为主要输出
				z = z[-1, :, :].view(1, local_bs, feats)
				
				# 三路输出的损失计算
				series, prior1, prior2 = output_list[0], output_list[1], output_list[2]
				# series = series[-1, :, :].view(1, local_bs, feats)
				# prior1 = prior1[-1, :, :].view(1, local_bs, feats)
				# prior2 = prior2[-1, :, :].view(1, local_bs, feats)
				
				# # 三路输出两两计算一致性损失（3个一致性损失）
				# consistency_loss_1 = l(series, prior1)  # series vs prior1
				# consistency_loss_2 = l(series, prior2)  # series vs prior2  
				# consistency_loss_3 = l(prior1, prior2)  # prior1 vs prior2
				
				# # 使用三个一致性损失的平均值作为辅助损失l2
				# l2 = 0.35*consistency_loss_1 + 0.35*consistency_loss_2 + 0.3*consistency_loss_3

				series = series.transpose(0, 1)
				prior1 = prior1.transpose(0, 1)
				prior2 = prior2.transpose(0, 1)

				consistency_loss_1 = compute_consistency_loss(series, prior1)
				consistency_loss_2 = compute_consistency_loss(series, prior2)
				consistency_loss_3 = compute_consistency_loss(prior1, prior2)
				
				# 使用三个一致性损失的平均值作为辅助损失l2
				l2 = consistency_loss_1 + consistency_loss_2 + consistency_loss_3
				l2 = torch.mean(l2)
				
				# 主要重构损失l1，采用ATF_UAD的损失耦合机制
				l1 = torch.mean(loss_func1(z, elem)) + 0.01*l2
				
				l1s.append(l1.item())
				l2s.append(l2.item())
				
				# 分步优化，类似ATF_UAD
				optimizer.zero_grad()
				# loss2.backward(retain_graph=True)  # 先优化辅助损失
				# loss1.backward()  # 再优化主损失
				# l2.backward(retain_graph=True)
				l1.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			# 测试阶段：收集所有批次的结果
			all_losses = []
			all_consistency_losses = []  # 新增：收集一致性损失
			all_z = []
			with torch.no_grad():
				for d, _ in dataloader:
					actual_bs = d.shape[0]  # 使用实际的batch size
					print(d.shape)
					window = d.permute(1, 0, 2)  # (seq_len, batch_size, features)
					elem = window[-1, :, :].view(1, actual_bs, feats)  # (1, batch_size, dim)
					z, output_list = model(window)
					# DC_UAD输出维度现在是 (window_size, batch_size, features)，与ATF_UAD一致
					z = z[-1, :, :].view(1, actual_bs, feats)  # 取最后一个时间步

					# 计算重构损失
					batch_loss = l(z, elem)[0]
					
					# 计算一致性损失（与训练阶段保持一致）
					series, prior1, prior2 = output_list[0], output_list[1], output_list[2]
					series = series.transpose(0, 1)  # (batch_size, seq_len, features)
					prior1 = prior1.transpose(0, 1)
					prior2 = prior2.transpose(0, 1)
					
					consistency_loss_1 = compute_consistency_loss(series, prior1)
					consistency_loss_2 = compute_consistency_loss(series, prior2)
					
					# 合并一致性损失（取平均值，与训练时保持一致）
					batch_consistency_loss = (consistency_loss_1 + consistency_loss_2) / 2.0
					
					all_losses.append(batch_loss.detach().cpu().numpy())
					all_consistency_losses.append(batch_consistency_loss.detach().cpu().numpy())
					all_z.append(z.detach().cpu().numpy()[0])
			
			# 拼接所有批次的结果
			loss = np.concatenate(all_losses, axis=0)
			consistency_loss = np.concatenate(all_consistency_losses, axis=0)
			z_pred = np.concatenate(all_z, axis=0)
			return loss, z_pred, consistency_loss  # 返回重构损失、预测值和一致性损失
			# with torch.no_grad():
			# 	for d, _ in dataloader:
			# 		window = d.permute(1, 0, 2)
			# 		local_bs = d.shape[0]
			# 		elem = window[-1, :, :].view(1, local_bs, feats)
			# 		z, _ = model(window)
			# 		z = z[-1, :, :].view(1, local_bs, feats)
			# 		loss = l(z, elem)[0]
			# return loss.detach().numpy(), z.detach().numpy()[0]
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()

def compute_consistency_loss(seq1, seq2, temperature=0.5):
	bs, seq_len, feature_dim = seq1.shape
	seq1_flat = F.normalize(seq1.reshape(-1, feature_dim), dim=1)
	seq2_flat = F.normalize(seq2.reshape(-1, feature_dim), dim=1)
	
	similarity = torch.matmul(seq1_flat, seq2_flat.T) / temperature

	mask = torch.eye(bs * seq_len, device=seq1.device)
	valid_similarity = similarity * mask
	
	loss = F.log_softmax(valid_similarity, dim=1)
	loss = -loss.diag()
	return loss.reshape(bs, seq_len).mean(dim=1)

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)
	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'ATF_UAD', 'DC_UAD'] or 'TranAD' in model.name:
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 10; e = epoch + 1; start = time()
		early_stopping = EarlyStopping(patience=3, verbose=True)
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			early_stopping(lossT, model, optimizer, scheduler, e, accuracy_list)
			if early_stopping.early_stop:
				print("Early stopping")
				break
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	loss, y_pred, consistency_loss = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

	# ### Plot curves
	# if not args.test:
	# 	if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0)
	# 	plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

	### Scores
	df = pd.DataFrame()
	preds = []  # Initialize preds list
	lossT, _, consistencyT = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result, pred = pot_eval(lt, l, ls); preds.append(pred)
		df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)

	# 计算最终的损失分数
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	# 一致性损失已经是每个样本的标量值，不需要在axis=1上求平均
	consistencyTfinal, consistencyFinal = consistencyT, consistency_loss
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0

	# 策略1：仅使用重构损失（原始方法）
	result_recon, label_pred_recon = pot_eval(lossTfinal, lossFinal, labelsFinal)
	
	# 策略2：集成OR策略（任一种损失判定为异常就认为是异常）
	# 首先需要计算一致性损失的预测结果
	result_consistency, label_pred_consistency = pot_eval(consistencyTfinal, consistencyFinal, labelsFinal)
	label_pred_ensemble_or = np.logical_or(label_pred_recon, label_pred_consistency).astype(int)
	
	# 计算各种策略的性能指标
	from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
	
	def compute_metrics(y_true, y_pred, scores):
		precision = precision_score(y_true, y_pred, zero_division=0)
		recall = recall_score(y_true, y_pred, zero_division=0)
		f1 = f1_score(y_true, y_pred, zero_division=0)
		try:
			auc = roc_auc_score(y_true, scores)
		except:
			auc = 0.0
		return {'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}
	
	# 只使用集成OR策略
	metrics_ensemble_or = compute_metrics(labelsFinal, label_pred_ensemble_or, lossFinal)
	
	# 使用集成OR策略作为最终结果
	result = {
		'precision': metrics_ensemble_or['precision'],
		'recall': metrics_ensemble_or['recall'],
		'f1': metrics_ensemble_or['f1'],
		'ROC/AUC': metrics_ensemble_or['auc'],
		'threshold': result_recon.get('threshold', 0)  # 添加重构损失的threshold
	}
	label_pred = label_pred_ensemble_or
	
	# 打印集成OR策略的性能
	print("\n集成OR策略性能:")
	print("=" * 80)
	print(f"ensemble_or     | F1: {metrics_ensemble_or['f1']:.4f} | Precision: {metrics_ensemble_or['precision']:.4f} | Recall: {metrics_ensemble_or['recall']:.4f} | AUC: {metrics_ensemble_or['auc']:.4f}")
	print("=" * 80)
	
	# 使用重构损失分数作为异常分数的代表
	best_scores = lossFinal
	best_train_scores = lossTfinal
	
	# # 添加策略信息到结果中
	# result['best_strategy'] = 'ensemble_or'
	# result['strategy_metrics'] = {'ensemble_or': metrics_ensemble_or}

	result.update(hit_att(loss, labels))
	result.update(ndcg(loss, labels))
	print(df)
	# print(result)

	### Save results to CSV
	# Create results directory if it doesn't exist
	results_dir = 'results'
	os.makedirs(results_dir, exist_ok=True)
	
	# Save detailed data for plotting
	plot_data = {
		'dataset': args.dataset,
		'model': args.model,
		'anomaly_scores': result.get('anomaly_scores', best_scores.tolist()),  # 使用重构损失分数作为异常分数
		'train_scores': best_train_scores.tolist(),   # 使用重构损失分数作为训练分数
		'true_labels': labelsFinal.tolist(),   # True binary labels
		'predictions': label_pred.tolist(),    # Predicted binary labels (集成OR策略的预测结果)
		'threshold': result.get('threshold', 0),
		'test_data': testO.cpu().numpy().tolist() if hasattr(testO, 'cpu') else testO.tolist(),  # Original test data
		'metrics': result
	}
	
	# Save plot data to pickle file
	plot_data_filename = os.path.join(results_dir, f'{args.dataset}_{args.model}_plot_data.pkl')
	with open(plot_data_filename, 'wb') as f:
		pickle.dump(plot_data, f)
	print(f'{color.GREEN}Plot data saved to {plot_data_filename}{color.ENDC}')
	
	# Prepare result data for saving
	result_data = {
		'Dataset': [args.dataset],
		'Model': [args.model],
		'Best_Strategy': ['ensemble_or'],  # 固定为集成OR策略
		'FN': [result.get('FN', 0)],
		'FP': [result.get('FP', 0)],
		'TN': [result.get('TN', 0)],
		'TP': [result.get('TP', 0)],
		'precision': [result.get('precision', 0)],
		'recall': [result.get('recall', 0)],
		'f1': [result.get('f1', 0)],
		'ROC/AUC': [result.get('ROC/AUC', 0)],
		'Hit@100%': [result.get('Hit@100%', 0)],
		'Hit@150%': [result.get('Hit@150%', 0)],
		'NDCG@100%': [result.get('NDCG@100%', 0)],
		'NDCG@150%': [result.get('NDCG@150%', 0)],
		'threshold': [result.get('threshold', 0)],
	}
	
	# Create DataFrame and save to CSV
	result_df = pd.DataFrame(result_data)
	csv_filename = os.path.join(results_dir, f'{args.dataset}_results.csv')
	
	# Check if file exists to decide whether to append or create new
	if os.path.exists(csv_filename):
		# Read existing data and append new results
		existing_df = pd.read_csv(csv_filename)
		combined_df = pd.concat([existing_df, result_df], ignore_index=True)
		combined_df.to_csv(csv_filename, index=False)
		print(f'{color.GREEN}Results appended to {csv_filename}{color.ENDC}')
	else:
		# Create new file
		result_df.to_csv(csv_filename, index=False)
		print(f'{color.GREEN}Results saved to {csv_filename}{color.ENDC}')
	
	print(f'{color.BOLD}Test results summary:{color.ENDC}')
	print(f'Dataset: {args.dataset}')
	print(f'Model: {args.model}')
	print(f'F1-Score: {result.get("f1", 0):.6f}')
	print(f'ROC/AUC: {result.get("ROC/AUC", 0):.6f}')
	print(f'Precision: {result.get("precision", 0):.6f}')
	print(f'Recall: {result.get("recall", 0):.6f}')
