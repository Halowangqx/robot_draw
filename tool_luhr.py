      
import torch
import builtins
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
import zipfile
from torchinfo import summary
from torchvision.transforms import transforms
import json, requests

def train_loop(model, dataloader, epoch,lr=0.01, optimizer=None, loss_fn=nn.NLLLoss(), device='cuda',report=100):
	'''
	:param model: network
	:param dataloader:train_dataloader
	:param lr:learning rate default=1e-2
	:param optimizer:
	:param loss_fn:default nn.NLLLoss()
	:param device:default cuda
	:return:(avgloss,avgaccuracy),the avgloss and avgaccuracy in this train_epoch

	careful:the loss present here is calculated as follows
	batchloss=loss_in_batch/batch_len
	avgloss=totalloss/sizeof(dataloader.dataset)
	avgaccuracy=totalcorrect/sizeof(dataloader.dataset)
	'''
	optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
	model.train()
	total_loss=0
	size = len(dataloader.dataset)
	for batch, (features, labels) in enumerate(dataloader):
		pred = model(features.float().to(device))
		loss = loss_fn(pred, labels.float().to(device))
		#print(labels.shape)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		#print(loss.shape)
		total_loss += loss.item()

		if batch % report == 0:  # 每一百个batch报告一次
			currentloss = loss.item() / len(features)
			current = (batch+1) * len(features)
			print("Epoch {}, minibatch {}:train loss = {} [current={}/size={}]".format(epoch, batch,total_loss/ current,current,size))
	
	return total_loss / size


def test_loop(model, dataloader, epoch,loss_fn=nn.NLLLoss(), device='cuda'):
	'''
	:param model: network
	:param dataloader: test_dataloader
	:param loss_fn: loss_function default:nn.NLLLoss()
	:param device:default cuda
	:return: (avgloss,avgaccuracy)
	'''
	model.eval()  # 切换到test模式
	acc, loss = 0, 0
	size = len(dataloader.dataset)
	with torch.no_grad():
		for features, labels in dataloader:
			out = model(features.to(device))
			loss += loss_fn(out, labels.to(device))
	return loss.item() / size


def train(net, train_loader, test_loader, optimizer=None, lr=0.01, epochs=10, loss_fn=nn.NLLLoss(),report=100,writer=None):
	'''

	:param net:
	:param train_loader:
	:param test_loader:
	:param optimizer:
	:param lr: default 0.01
	:param epochs: default 10
	:param loss_fn: default NLLLoss()
	:return:a dict
	'''
	optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
	for ep in range(epochs):
		print(f"Epoch {ep + 1}\n-------------------------------")
		tl= train_loop(model=net, dataloader=train_loader, optimizer=optimizer, lr=lr, loss_fn=loss_fn,epoch=ep+1,report=report)
		vl= test_loop(model=net, dataloader=test_loader,epoch=ep+1,loss_fn=loss_fn)
		print(
			f"Epoch {ep + 1:2},Train loss={tl:.3f}, Val loss={vl:.3f}")


def plot_results(hist):
	'''
	can be used with train
	:param hist: a dict
	:return: the curve of avgloss and avgacc of each epoch
	'''
	plt.figure(figsize=(15, 5))
	plt.subplot(121)
	plt.plot(hist['train_acc'], label='Training acc')
	plt.plot(hist['val_acc'], label='Validation acc')
	plt.legend()
	plt.subplot(122)
	plt.plot(hist['train_loss'], label='Training loss')
	plt.plot(hist['val_loss'], label='Validation loss')
	plt.legend()
	plt.show()
	

def check_image(fn):
	try:
		im = Image.open(fn)
		im.verify()
		return True
	except:
		return False


def check_image_dir(path):
	for fn in glob.glob(path):
		if not check_image(fn):
			print("Corrupt image: {}".format(fn))
			os.remove(fn)

    