import subprocess
import argparse
import sys # 建议导入 sys 以便处理异常时的 flush

def main(args):
    # 定义工作目录
    work_dir = "/data/wlf/RAE-IGD-DMVAE"

    # 构建命令
    command = [
        "python", "IGD/train.py",
        "-d", "imagenet",
        "--imagenet_dir", args.imagenet_path,
        "-n", "resnet",
        "--depth", "18",
        "--nclass", "10",
        "--norm_type", "instance",
        "--ipc", "10",
        "--tag", "test",
        "--slct_type", "random",
        "--spec", "nette",
        "--batch_size", "32",
        "--verbose"
    ]

    print(f"切换工作目录至: {work_dir}")
    print("开始训练...")

    # 使用 Popen 启动进程
    process = subprocess.Popen(
        command,
        cwd=work_dir,             # <--- 关键点：在这里指定工作目录
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 实时输出
    for line in process.stdout:
        print(line, end='')

    # 等待进程结束
    return_code = process.wait()

    if return_code == 0:
        print("\n训练成功完成！")
    else:
        print(f"\n训练出错，返回码: {return_code}")


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='/data2/wlf/fractalgen-distillation/nette', type=str)
    parser.add_argument('--imagenet_path', default="/data/wlf/datasets/imagenette/", type=str)

    args = parser.parse_args()

    main(args)