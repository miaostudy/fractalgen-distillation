import os
import torch
import numpy as np
from models import fractalgen
from torchvision.utils import save_image
from util import download
from PIL import Image
from IPython.display import display
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_type = "fractalmar_huge_in256" #@param ["fractalmar_base_in256", "fractalmar_large_in256", "fractalmar_huge_in256"]
num_conds = 5 # 生成时条件嵌入数量

# 模型
model = fractalgen.__dict__[model_type](
    guiding_pixel=True, # ？
    num_conds=num_conds
).to(device)

state_dict = torch.load("pretrained_models/{}/checkpoint-last.pth".format(model_type))["model"]
model.load_state_dict(state_dict)
model.eval()

# 参数
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
np.random.seed(seed)
num_iter_list = 64, 16, 16 #@param {type:"raw"} #  迭代步数
cfg_scale = 10 #@param {type:"slider", min:1, max:20, step:0.5} # 类别标签对生成图像的约束强度
cfg_schedule = "constant" #@param ["linear", "constant"] # constant: 使用固定的cfg_scale
temperature = 1.1 #@param {type:"slider", min:0.9, max:1.2, step:0.01} # 随机性
filter_threshold = 1e-3 # 过滤阈值
class_labels = [207] #@param {type:"raw"}
samples_per_row = 1 #@param {type:"number"} # 每行展示的数量

label_gen = torch.Tensor(class_labels).long().cuda()
class_embedding = model.class_emb(label_gen)

if not cfg_scale == 1.0:
  class_embedding = torch.cat([class_embedding, model.fake_latent.repeat(label_gen.size(0), 1)], dim=0)

with torch.no_grad():
  with torch.cuda.amp.autocast():
    sampled_images = model.sample(
      cond_list=[class_embedding for _ in range(num_conds)],
      num_iter_list=num_iter_list,
      cfg=cfg_scale, cfg_schedule=cfg_schedule,
      temperature=temperature,
      filter_threshold=filter_threshold,
      fractal_level=0,
      save_path='test',
      visualize=True
    )

# Denormalize images.
pix_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(1, -1, 1, 1)
pix_std = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(1, -1, 1, 1)
sampled_images = sampled_images * pix_std + pix_mean
sampled_images = sampled_images.detach().cpu()

# Save & display images
save_image(sampled_images, "samples.png", nrow=int(samples_per_row), normalize=True, value_range=(0, 1))
samples = Image.open("samples.png")