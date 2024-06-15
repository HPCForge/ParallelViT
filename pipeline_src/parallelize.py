import torch
from torch import nn
from transformers import ViTConfig, ViTModel
import torchsummary
from collections import OrderedDict
import numpy as np
import torchsummary

from vit import SelfAttentionBlock, PatchEmbed, ViT
from afno import AFNONet
import time


class ParallelViT():

	def __init__(self, model, freeze_weights=True):

		self.buffer_size = 1024 #in MB

		self.model = model	
		self.blocks = []
		
		self.num_devices = torch.cuda.device_count()
		self.get_cuda_devices()


		self.collect_blocks(model)
		#self.print_blocks()

		if freeze_weights:
			self.freeze_weights()
		
		self.print_blocks()
		self.partition_blocks()
	
		self.get_total_model_size()

	def get_cuda_devices(self):
		self.devices = []
		for i in range(self.num_devices):

			total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
			reserved_memory = torch.cuda.memory_reserved(i) / (1024 ** 2)
			available_memory = total_memory - reserved_memory - self.buffer_size # in megabytes
			
			self.devices.append({"device_id": i, "device_name": torch.cuda.get_device_name(i), "available_memory": available_memory})


	def collect_blocks(self, module, prefix=''):
		for name, sub_module in module.named_children():
			full_name = f'{prefix}.{name}' if prefix else name
			if isinstance(sub_module, (nn.ModuleList, nn.Sequential)):
				for idx, sm in enumerate(sub_module):
					self.collect_blocks(sm, f'{full_name}.{idx}')
			else:

				total_params = sum(param.numel() * param.element_size() for param in sub_module.parameters())
				block_size = total_params / (1024 ** 2) #megabytes
				self.blocks.append({"name": full_name, "module": sub_module, "param_count": total_params, "block_size": block_size})
		

	def freeze_weights(self):
		for block in self.blocks:
			for param in block["module"].parameters():
				param.requires_grad = False
	
	def print_blocks(self):

		for block in self.blocks:
			print(block)
	
	def partition_blocks(self):
		if torch.cuda.is_available():
			self.allocation = []
			gpu_index = 0
			for block in self.blocks:
				block_name = block['name']
				block_size = block['block_size']

				grad_size = block['param_count'] / (1024 **2 )
				activation_size = ((16 * 2048 * 1024) * 4) / (1024 ** 2)

				total_required_memory = block_size + grad_size + activation_size

				while gpu_index < len(self.devices):
					device = self.devices[gpu_index]
					device_id = device['device_id']
					gpu_memory = device['available_memory']

					if block_size <= gpu_memory:
						self.allocation.append({'block_name': block_name, 'device_id': device_id, 'block_size': block_size})
						self.devices[gpu_index]['available_memory'] -= total_required_memory
						break
					else:
						gpu_index += 1

				if gpu_index >= len(self.devices):
					raise RuntimeError("Not enough GPU memory to allocate all blocks.")
			
			for i, alloc in enumerate(self.allocation):
				print(f"Block {alloc['block_name']} is allocated to GPU {alloc['device_id']} with size {alloc['block_size']} MB")
				self.blocks[i]["module"] = self.blocks[i]["module"].to(f"cuda:{alloc['device_id']}")

		else:
			print("CUDA is not available")

		
	def get_total_model_size(self):
		total_size = 0
		for block in self.blocks:
			total_size += block["block_size"]
		
		print(f"total size: {total_size / 1024} GB")
	
	def forward(self, x):
		
		for i, block in enumerate(self.blocks):
			device_id = self.allocation[i]["device_id"]
			device = torch.device(f"cuda:{device_id}")
			x = x.to(device)
			x = block["module"](x)
			
		return x
			
		

def main():

	field_height = 512
	field_width = 512
	patch_size = 16
	pred_channels = 1
	embed_dim = 1024
	depth = 72
	num_heads = 16
	batch_size = 16
	input_shape = (batch_size, pred_channels, field_height, field_width)

	#model = AFNONet(field_height=field_height, field_width=field_width, patch_size=patch_size, pred_channels=pred_channels, embed_dim=embed_dim, depth=depth)
	model = ViT(field_height=field_height, field_width=field_width, patch_size=patch_size, pred_channels=pred_channels, embed_dim=embed_dim, depth=depth)
	total_params = sum(p.numel() for p in model.parameters())
	print(total_params)
	parallel_model = ParallelViT(model)

	rand_input = torch.rand(input_shape)
	print(f"input shape: {rand_input.shape}")
	
	start_time = time.time()
	output = parallel_model.forward(rand_input)
	end_time = time.time()

	print(f"output shape: {output.shape}")

	print(end_time-start_time)
	

if __name__ == "__main__":

	main()