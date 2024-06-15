from transformers import ViTConfig, ViTModel
import torch

from parallelformers import parallelize
import time
import torchsummary

from src.vit import ViT

device_count = torch.cuda.device_count()
print(f"running on {device_count} gpus")

def model_size_mb(model):
	total_params = sum(p.numel() for p in model.parameters())
	total_size_bytes = total_params * model.parameters().__next__().element_size()
	total_size_MB = total_size_bytes / (1024 ** 2)
	return total_size_MB

def main():
	# Configuration for the Vision Transformer

	img_size = 512
	
	config = ViTConfig(
		image_size=img_size,
		patch_size=16,		
		num_channels=1,	
		num_hidden_layers=72,
		num_attention_heads=16,
		hidden_size=1024,
		intermediate_size=4096,
		hidden_act="gelu",
		hidden_dropout_prob=0.1,
		attention_probs_dropout_prob=0.1,
	)

	
	model = ViTModel(config)
	total_params = sum(p.numel() for p in model.parameters())
	print(total_params)

	parallelize(model, num_gpus=device_count, fp16=True, verbose='detail')	



	batch_size = 16
	random_inputs = torch.rand(batch_size, 1, img_size, img_size)  # Random data simulating [batch_size, channels, height, width]

	start_time = time.time()
	output = model(random_inputs)
	end_time = time.time()
	print(end_time-start_time)



if __name__ == "__main__":
	main()
