import os
import torch
import numpy as np
from models import fractalgen
from torchvision.utils import save_image
from util import download
from PIL import Image
import argparse

class_labels = [0,217,482,491,497,566,569,571,574,701]

model_type = "fractalmar_huge_in256"
num_conds = 1 # 生成时条件嵌入数量

# 参数
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
np.random.seed(seed)
num_iter_list = 64, 16, 16 #@param {type:"raw"} #  迭代步数
cfg_scale = 10 #@param {type:"slider", min:1, max:20, step:0.5} # 类别标签对生成图像的约束强度
cfg_schedule = "constant" #@param ["linear", "constant"] # constant: 使用固定的cfg_scale
temperature = 1.1 #@param {type:"slider", min:0.9, max:1.2, step:0.01} # 随机性
filter_threshold = 1e-3 # 过滤阈值

def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    # 模型
    model = fractalgen.__dict__[model_type](
        guiding_pixel=True,
        num_conds=num_conds
    ).to(device)

    state_dict = torch.load("pretrained_models/{}/checkpoint-last.pth".format(model_type))["model"]
    model.load_state_dict(state_dict)
    model.eval()

    class_names = []
    with open('misc/class_nette.txt', 'r') as f:
        lines = f.readline()
        for line in lines:
            class_names.append(line.strip())

    print("Number of classes: {}".format(len(class_names)))

    # 逐个生成每个类的图像/可以试试一起生成然后切分
    for i, label in enumerate(class_labels):
        class_embedding = model.class_emb([label])
        label_gen = torch.Tensor([label]).long().cuda()

        if not cfg_scale == 1.0:
            class_embedding = torch.cat([class_embedding, model.fake_latent.repeat(label_gen.size(0), 1)], dim=0)

        save_dir = os.path.join(args.save_dir, class_names[i])
        os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                sampled_images = model.sample(
                    cond_list=[class_embedding for _ in range(num_conds)],
                    num_iter_list=num_iter_list,
                    cfg=cfg_scale, cfg_schedule=cfg_schedule,
                    temperature=temperature,
                    filter_threshold=filter_threshold,
                    save_dir=save_dir,
                    fractal_level=0,
                    visualize=True
                )

        pix_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(1, -1, 1, 1)
        pix_std = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(1, -1, 1, 1)
        sampled_images = sampled_images * pix_std + pix_mean
        sampled_images = sampled_images.detach().cpu()

        save_image(sampled_images, "samples.png", nrow=1, normalize=True, value_range=(0, 1))

if __name__ == 'main':
    args = argparse.ArgumentParser()
    args.add_argument("--save_dir", default='results/imagenette', type=str)

    parsed_args = args.parse_args()
    main(parsed_args)
