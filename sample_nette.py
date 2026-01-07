import os
import torch
import numpy as np
from PIL import Image
from models import fractalgen
from util import download
import argparse
from tqdm import tqdm


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.class_names_path, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    with open(args.class_indices_path, 'r') as f:
        class_indices = [int(line.strip()) for line in f if line.strip()]

    assert len(class_names) == len(class_indices), "类名文件和索引文件行数不匹配！"

    print(f"Loading model: {args.model_type}...")
    download_func = getattr(download, f"download_pretrained_{args.model_type}", None)
    if download_func: download_func()

    model = fractalgen.__dict__[args.model_type](guiding_pixel=True, num_conds=1).to(device)
    state_dict = torch.load(f"pretrained_models/{args.model_type}/checkpoint-last.pth")["model"]
    model.load_state_dict(state_dict)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    num_iter_list = [int(x) for x in args.steps.split(',')]

    print(f"Start sampling {args.ipc} images per class for {len(class_names)} classes...")

    for c_idx, (class_name, class_idx_int) in tqdm(enumerate(zip(class_names, class_indices))):
        if class_idx_int == -1: continue

        print(f"[{c_idx + 1}/{len(class_names)}] Class: {class_name} (Index: {class_idx_int})")

        num_batches = (args.ipc + args.batch_size - 1) // args.batch_size
        generated_count = 0

        for b in range(num_batches):
            current_bsz = min(args.batch_size, args.ipc - generated_count)

            labels = torch.tensor([class_idx_int] * current_bsz).long().to(device)
            class_embedding = model.class_emb(labels)

            if args.cfg != 1.0:
                fake_latent = model.fake_latent.repeat(current_bsz, 1)
                class_embedding = torch.cat([class_embedding, fake_latent], dim=0)

            cond_list = [class_embedding for _ in range(model.num_conds)]

            def save_step_callback(patches_tensor, step):
                """
                patches_tensor: (B, 3, H, W) 或者是 (2*B, ...) 如果开了 CFG
                step: 当前步数 (0-indexed)
                """
                # 如果开了 CFG，patches_tensor 包含了 cond 和 uncond 的结果拼接
                # 通常上半部分是 cond，下半部分是 uncond (取决于实现，这里通常前面是 cond)
                # MAR 代码里: torch.cat([patches, patches], dim=0) -> 前半部分是原 patches
                if args.cfg != 1.0:
                    real_bsz = patches_tensor.shape[0] // 2
                    patches_tensor = patches_tensor[:real_bsz]

                # 反归一化
                imgs = patches_tensor * std + mean
                imgs = torch.clamp(imgs, 0, 1)
                imgs_np = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()  # (B, H, W, 3)
                imgs_np = (imgs_np * 255).astype(np.uint8)

                # 保存路径结构: save_root / step / class_name / img_id.png
                step_dir = os.path.join(args.save_root, str(step + 1))  # step 从 1 开始
                class_dir = os.path.join(step_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                for i in range(current_bsz):
                    img_id = generated_count + i
                    save_path = os.path.join(class_dir, f"{img_id}.png")
                    Image.fromarray(imgs_np[i]).save(save_path)

            # --- 执行采样 ---
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    model.sample(
                        cond_list=cond_list,
                        num_iter_list=num_iter_list,
                        cfg=args.cfg,
                        cfg_schedule="constant",
                        temperature=1.0,
                        filter_threshold=1e-3,
                        fractal_level=0,
                        visualize=False,
                        step_callback=save_step_callback  # <--- 传入回调
                    )

            generated_count += current_bsz

    print(f"Done! Results saved to {args.save_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='fractalmar_huge_in256', type=str)
    parser.add_argument('--ipc', default=10, type=int, help="Images Per Class (每类采样多少张)")
    parser.add_argument('--save_root', default='intermediate_samples', type=str)
    parser.add_argument('--class_names_path', default='misc/class_nette.txt', type=str, help="类名列表文件")
    parser.add_argument('--class_indices_path', default='class_indices.txt', type=str, help="类索引文件")

    # 生成参数
    parser.add_argument('--steps', default="64,16,16", type=str)
    parser.add_argument('--cfg', default=4.0, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=4, type=int)

    args = parser.parse_args()
    main(args)