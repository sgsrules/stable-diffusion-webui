# Author: thygate
# https://github.com/thygate/stable-diffusion-webui-depthmap-script

import modules.scripts as scripts
import gradio as gr
from einops import rearrange, repeat
from modules import processing, images, shared, sd_samplers, devices
from modules.processing import create_infotext, process_images, Processed
from modules.shared import opts, cmd_opts, state, Options
from PIL import Image

import torch, gc
import cv2
import requests
import os.path
import contextlib

from torchvision.transforms import Compose
from repositories.midas.midas.dpt_depth import DPTDepthModel
from repositories.midas.midas.midas_net import MidasNet
from repositories.midas.midas.midas_net_custom import MidasNet_small
from repositories.midas.midas.transforms import Resize, NormalizeImage, PrepareForNet

import numpy as np
#import matplotlib.pyplot as plt

class MidasDepth():
	def InitMidas(self, model_type):
		def download_file(filename, url):
			print("Downloading midas model weights to %s" % filename)
			with open(filename, 'wb') as fout:
				response = requests.get(url, stream=True)
				response.raise_for_status()
				# Write response data to file
				for block in response.iter_content(4096):
					fout.write(block)

		# init torch device
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print("device: %s" % self.device)
		# model path and name
		model_dir = "./models/midas"
		# create path to model if not present
		os.makedirs(model_dir, exist_ok=True)
		print("Loading midas model weights from ", end=" ")
		# "dpt_large"
		if model_type == 0:
			model_path = f"{model_dir}/dpt_large-midas-2f21e586.pt"
			print(model_path)
			if not os.path.exists(model_path):
				download_file(model_path,
							  "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt")
			self.model = DPTDepthModel(
				path=model_path,
				backbone="vitl16_384",
				non_negative=True,
			)
			net_w, net_h = 384, 384
			self.resize_mode = "minimal"
			self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

		# "dpt_hybrid"
		elif model_type == 1:
			model_path = f"{model_dir}/dpt_hybrid-midas-501f0c75.pt"
			print(model_path)
			if not os.path.exists(model_path):
				download_file(model_path,
							  "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt")
			self.model = DPTDepthModel(
				path=model_path,
				backbone="vitb_rn50_384",
				non_negative=True,
			)
			net_w, net_h = 384, 384
			self.resize_mode = "minimal"
			self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

		# "midas_v21"
		elif model_type == 2:
			model_path = f"{model_dir}/midas_v21-f6b98070.pt"
			print(model_path)
			if not os.path.exists(model_path):
				download_file(model_path,
							  "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt")
			self.model = MidasNet(model_path, non_negative=True)
			net_w, net_h = 384, 384
			self.resize_mode = "upper_bound"
			self.normalization = NormalizeImage(
				mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
			)

		# "midas_v21_small"
		elif model_type == 3:
			model_path = f"{model_dir}/midas_v21_small-70d6b9c8.pt"
			print(model_path)
			if not os.path.exists(model_path):
				download_file(model_path,
							  "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt")
			self.model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
								   non_negative=True, blocks={'expand': True})
			net_w, net_h = 256, 256
			self.resize_mode = "upper_bound"
			self.normalization = NormalizeImage(
				mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
			)
		self.model.eval()
		# optimize
		if self.device == torch.device("cuda"):
			self.model = self.model.to(memory_format=torch.channels_last)
			if not cmd_opts.no_half:
				self.model = self.model.half()
		self.model.to( self.device)

	device = None
	model = None
	normalization = None
	resize_mode = None
	def GetDepth(self, inputImg,width,height,zscale,zoffset,invert_depth = True):

		# override net size
		net_width, net_height = width, height

		# init transform
		transform = Compose(
			[
				Resize(
					net_width,
					net_height,
					resize_target=None,
					keep_aspect_ratio=True,
					ensure_multiple_of=32,
					resize_method=self.resize_mode,
					image_interpolation_method=cv2.INTER_CUBIC,
				),
				self.normalization,
				PrepareForNet(),
			]
		)


		# input image
		#img = cv2.cvtColor(np.asarray(inputImg), cv2.COLOR_BGR2RGB) / 255.0
		img = inputImg.astype(np.float32) / 255.0
		img_input = transform({"image": img})["image"]

		# compute
		precision_scope = torch.autocast if shared.cmd_opts.precision == "autocast" and self.device == torch.device(
			"cuda") else contextlib.nullcontext
		with torch.no_grad(), precision_scope("cuda"):
			sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
			if self.device == torch.device("cuda"):
				sample = sample.to(memory_format=torch.channels_last)
				if not cmd_opts.no_half:
					sample = sample.half()
			prediction = self.model.forward(sample)
			prediction = (
				torch.nn.functional.interpolate(
					prediction.unsqueeze(1),
					size=img.shape[:2],
					mode="bicubic",
					align_corners=False,
				)
				.squeeze()
				.cpu()
				.numpy()
			)
		torch.cuda.empty_cache()
		# output
		depth = prediction
		#numbytes = 2
		mmin = depth.min()
		mmax = depth.max()
		#max_val = (2 ** (8 * numbytes)) - 1

		# check output before normalizing and mapping to 16 bit
		#if depth_max - depth_min > np.finfo("float").eps:
		#	out = (depth - depth_min) / (depth_max - depth_min)
		#	#out = max_val * (depth - depth_min) / (depth_max - depth_min)
		#else:
		#	out = np.zeros(depth.shape)

		#if invert_depth:
		#	out = 1.0-out

		out = (depth - mmin) / (mmax- mmin)
		out = 1 - out
		out = out * zscale + zoffset

		out = np.expand_dims(out, axis=0)
		depth_tensor = torch.from_numpy(out).squeeze().to(self.device)
		print(f"  depth min:{mmin} max:{mmax} out min:{out.min()} max: {out.max()}")
		return depth_tensor
		# single channel, 16 bit image
		#img_output = out.astype("uint16")

		# invert depth map
		#if invert_depth:
		#	img_output = cv2.bitwise_not(img_output)

		#return Image.fromarray(img_output)

	def save(filename: str, depth: torch.Tensor):
		depth = depth.cpu().numpy()
		if len(depth.shape) == 2:
			depth = np.expand_dims(depth, axis=0)
		temp = rearrange(depth/depth.max() * 255, 'c h w -> h w c')
		temp = repeat(temp, 'h w 1 -> h w c', c=3)
		Image.fromarray(temp.astype(np.uint8)).save(filename)