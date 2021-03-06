#!/usr/bin/env python3

# Author: Nathan Inkawhich <https://github.com/inkawhich>

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pyprof
import torch.cuda.nvtx as nvtx
pyprof.init()

pretrained_model = "lenet_mnist_model.pth"
use_cuda=True

# LeNet Model definition
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=False, download=True,
		transform=transforms.Compose([transforms.ToTensor(),])),
	batch_size=1, shuffle=False)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
	# Collect the element-wise sign of the data gradient
	sign_data_grad = data_grad.sign()

	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_image = image + epsilon*sign_data_grad

	# Adding clipping to maintain [0,1] range
	perturbed_image = torch.clamp(perturbed_image, 0, 1)

	# Return the perturbed image
	return perturbed_image

def test(model, device, test_loader, epsilon):

	adv_examples = []

	# Loop over all examples in test set
	for data, target in test_loader:

		# Send the data and label to the device
		data, target = data.to(device), target.to(device)

		# Set requires_grad attribute of tensor. Important for Attack
		data.requires_grad = True

		# Forward pass the data through the model
		output = model(data)

		# Get the index of the max log-probability
		init_pred = output.max(1, keepdim=True)[1]

		# If the initial prediction is wrong, dont bother attacking
		if init_pred.item() != target.item():
			continue

		# Calculate the loss
		loss = F.nll_loss(output, target)

		# Zero all existing gradients
		model.zero_grad()

		# Calculate gradients of model in backward pass
		loss.backward()

		# Collect datagrad
		data_grad = data.grad.data

		# Call FGSM Attack
		perturbed_data = fgsm_attack(data, epsilon, data_grad)

		# Re-classify the perturbed image
		output = model(perturbed_data)

		# Check for success
		final_pred = output.max(1, keepdim=True)[1]
		if final_pred.item() != target.item():
			# Save some examples for visualization later
			if len(adv_examples) < 5:
				adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
				adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
			else:
				break

	# Return the adversarial examples
	return adv_examples

eps = .3
with torch.autograd.profiler.emit_nvtx():
	examples = test(model, device, test_loader, eps)

plt.figure(figsize=(10,10))
for i in range(len(examples)):
	plt.subplot(1,len(examples),i+1)
	plt.xticks([], [])
	plt.yticks([], [])
	orig,adv,ex = examples[i]
	plt.title("{} -> {}".format(orig, adv))
	#plt.imshow(ex, cmap="gray")
	plt.imshow(ex)
#plt.show()
plt.savefig("output.png", bbox_inches='tight')
