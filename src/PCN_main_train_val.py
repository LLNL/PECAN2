################################################################################
# Copyright 2023-2024 Lawrence Livermore National Security, LLC and other
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Pose Classification with 3D Atomic Network 2 (PECAN2)
# PCN main train validation
################################################################################
import os
import sys
sys.stdout.flush()
sys.path.insert(0, "../common")
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, RMSprop, lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset

from PCN_model_pcn import Model_PCN, Model_Transform, strip_prefix_if_present
from PCN_data_reader import Dataset_MLHDF
from file_util import *


# program arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device-name", default="cuda:0", help="use cpu or cuda:0, cuda:1 ...")
parser.add_argument("--data-dir", default="", help="dataset directory")
parser.add_argument("--dataset-type", type=float, default=1, help="ml-hdf version, (1: for fusion, 1.5: for cfusion 2: ml-hdf v2)")
parser.add_argument("--mlhdf-fn", default="", help="training docking ml-hdf path")
parser.add_argument("--cmlhdf-fn", default="", help="training crystal ml-hdf path")
parser.add_argument("--csv-fn", default="", help="training csv file path")
parser.add_argument("--valcsv-fn", default="", help="validation csv file path")
parser.add_argument("--model-path", default="", help="model checkpoint file path")
parser.add_argument("--affine-trans", default=False, action="store_true", help="use affine transformation or not")
parser.add_argument("--use-feat", default=False, action="store_true", help="use ligand and interaction feature or not")
parser.add_argument("--max-atoms", type=int, default=1000, help="maximum number of atoms")
parser.add_argument("--epoch-count", type=int, default=50, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=50, help="mini-batch size")
parser.add_argument("--learning-rate", type=float, default=0.0007, help="initial learning rate")
parser.add_argument("--decay-rate", type=float, default=0.97, help="learning rate decay")
parser.add_argument("--decay-iter", type=int, default=1000, help="learning rate decay")
parser.add_argument("--checkpoint-iter", type=int, default=50, help="checkpoint save rate, if zero, then save only when loss decreases")
parser.add_argument("--multi-gpus", default=False, action="store_true", help="whether to use multi-gpus")
parser.add_argument("--verbose", type=int, default=0, help="print all input/output shapes or not")
args = parser.parse_args()


# set CUDA for PyTorch
use_cuda = torch.cuda.is_available()
cuda_count = torch.cuda.device_count()
if use_cuda:
	device = torch.device(args.device_name)
	torch.cuda.set_device(int(args.device_name.split(':')[1]))
else:
	device = torch.device("cpu")
print(use_cuda, cuda_count, device)



def worker_init_fn(worker_id):
	np.random.seed(int(0))

def train():

	# load dataset
	csv_path = os.path.join(args.data_dir, args.csv_fn)
	val_csv_path = os.path.join(args.data_dir, args.valcsv_fn)
	mlhdf_path = os.path.join(args.data_dir, args.mlhdf_fn)
	cmlhdf_path = os.path.join(args.data_dir, args.cmlhdf_fn)
	print(csv_path,val_csv_path, mlhdf_path, cmlhdf_path, args.use_feat, args.affine_trans)
	dataset = Dataset_MLHDF(csv_path, mlhdf_path, cmlhdf_path, feat_int=args.use_feat, affine_trans=args.affine_trans, max_atoms=args.max_atoms)
	val_dataset = Dataset_MLHDF(val_csv_path, mlhdf_path, cmlhdf_path, feat_int=args.use_feat, affine_trans=args.affine_trans, max_atoms=args.max_atoms)
	# check multi-gpus
	num_workers = 0
	if args.multi_gpus and cuda_count > 1:
		num_workers = cuda_count

	# initialize data loader
	batch_count = len(dataset) // args.batch_size
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=None, drop_last=True)
	
	# if validation set is available
	val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None, drop_last=True)

	# define model
	model = Model_PCN(input_dim=[args.max_atoms,22], use_feat=args.use_feat, verbose=args.verbose)
	model_t = Model_Transform(verbose=args.verbose)
	#if use_cuda:
	#	model = model.cuda()
	if args.multi_gpus and cuda_count > 1:
		model = nn.DataParallel(model)
	model.to(device)
	
	if isinstance(model, (DistributedDataParallel, DataParallel)):
		model_to_save = model.module
	else:
		model_to_save = model

	# set loss, optimizer, decay, other parameters
	loss_fn = nn.BCELoss().float()
	optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
	#optimizer = RMSprop(model.parameters(), lr=args.learning_rate)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_iter, gamma=args.decay_rate)

	# load model
	epoch_start = 0
	if valid_file(args.model_path):
		checkpoint = torch.load(args.model_path)
		model_state_dict = checkpoint.pop("model_state_dict")
		strip_prefix_if_present(model_state_dict, "module.")
		model_to_save.load_state_dict(model_state_dict, strict=False)
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		epoch_start = checkpoint["epoch"]
		loss = checkpoint["loss"]
		print("checkpoint loaded: %s" % args.model_path)

	if not os.path.exists(os.path.dirname(args.model_path)):
		os.makedirs(os.path.dirname(args.model_path))
	output_dir = os.path.dirname(args.model_path)
    
	best_val_acc = 0.0
	step = 0
	epoch_losses = []
	for epoch_ind in range(epoch_start, args.epoch_count):
 
        # training
		batch_losses = []
		model.train()
		for batch_ind, batch in enumerate(dataloader):

			# transfer to GPU
			x_cpu, xf_cpu, y_cpu = batch
			x = x_cpu.to(device)
			y = y_cpu.to(device)
			xf = xf_cpu.to(device)
			
			#x1 = x[:,:,:,:3]
			#x2 = x[:,:,:,3:]
			#print(x1.shape)
			#print(x2.shape)
			
			# forward training
			#trans = model_t(x1)
			#x1t = torch.matmul(torch.squeeze(x1), trans)
			#x1t = torch.unsqueeze(x1t, dim=1)
			#x1t2 = torch.cat([x1t, x2], 3)
			#yp, _ = model(x1t2)

			# forward model
			yp, _ = model(x, xf)

			# compute loss
			loss = loss_fn(yp.cpu().float(), y_cpu.float())
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			batch_loss = loss.cpu().data.item()
			batch_losses.append(batch_loss)
			print("[%d/%d-%d/%d] training, loss: %.3f, lr: %.7f" % (epoch_ind+1, args.epoch_count, batch_ind+1, batch_count, batch_loss, optimizer.param_groups[0]['lr']))
        
		epoch_loss = np.mean(batch_losses)
		epoch_losses.append(epoch_loss)
		print("[%d/%d] training, epoch loss: %.3f" % (epoch_ind+1, args.epoch_count, epoch_loss))
        
        # validation
		ytrue_arr = np.zeros((len(val_dataset),), dtype=np.float32)
		ypred_arr = np.zeros((len(val_dataset),), dtype=np.float32)
		model.eval()
		with torch.no_grad():
			for bind, batch in enumerate(val_dataloader):

                # transfer to GPU
				x_cpu, xf_cpu, y_cpu = batch
				x = x_cpu.to(device)
				y = y_cpu.to(device)
				xf = xf_cpu.to(device)
				
				# forward training
				yp, _ = model(x, xf)

				ytrue = y_cpu.float().data.numpy()[:,0]
				ypred = yp.cpu().float().data.numpy()[:,0]
				# if bind == batch_count:
				# 	ytrue_arr[bind*args.batch_size:len(dataset)] = ytrue
				# 	ypred_arr[bind*args.batch_size:len(dataset)] = ypred
				# else:
				ytrue_arr[bind*args.batch_size:(bind+1)*args.batch_size] = ytrue
				ypred_arr[bind*args.batch_size:(bind+1)*args.batch_size] = ypred

				print("[%d/%d] evaluating" % (bind+1, batch_count))
		ypred_arr2 = np.where(ypred_arr > 0.5, 1, 0)
		val_acc = 100.0 * np.sum(ypred_arr2 == ytrue_arr) / ytrue_arr.shape[0]
		print("accuracy is : ",val_acc)
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			checkpoint_dict = {
				"model_state_dict": model_to_save.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"loss": loss,
				"step": step,
				"epoch": epoch_ind
			}
			torch.save(checkpoint_dict, args.model_path)
			print("checkpoint saved: %s" % args.model_path)
			print("best validation accuracy is : ", best_val_acc)
		step += 1


#		if args.checkpoint_iter == 0 and (epoch_ind == 0 or epoch_loss < epoch_losses[-1]):
#			checkpoint_dict = {
#				"model_state_dict": model_to_save.state_dict(),
#				"optimizer_state_dict": optimizer.state_dict(),
#				"loss": loss,
#				"step": step,
#				"epoch": epoch_ind
#			}
#			torch.save(checkpoint_dict, args.model_path)
#			print("checkpoint saved: %s" % args.model_path)
		
		'''if val_dataset:
			val_losses = []
			model.eval()
			with torch.no_grad():
				for batch_ind, batch in enumerate(val_dataloader):
				
					x_cpu, y_cpu = batch
					x = x_cpu.to(device)
					y = y_cpu.to(device)

					yp, _ = model(x)
					loss = loss_fn(yp.cpu().float(), y.float())
						
					val_losses.append(loss.cpu().data.item())
					print("[%d/%d-%d/%d] validation, loss: %.3f" % (epoch_ind+1, args.epoch_count, batch_ind+1, batch_count, loss.cpu().data.item()))

				print("[%d/%d] validation, epoch loss: %.3f" % (epoch_ind+1, args.epoch_count, np.mean(val_losses)))'''

	# close dataset
	dataset.close()
	#if val_dataset != None:
	#	val_dataset.close()


def main():
	train()

if __name__ == "__main__":
	main()
