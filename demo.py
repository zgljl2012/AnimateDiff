"""

https://civitai.com/images/1858744

"""

import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image, schedulers, DPMSolverSDEScheduler

model_id = 'runwayml/stable-diffusion-v1-5'

pipe = AutoPipelineForText2Image.from_pretrained(model_id, use_safetensors=False, safety_checker=None)
