import subprocess
import argparse


def main(args):
    command1 = ["cd", "/data/wlf/RAE-IGD-DMVAE",]


    # 使用 Popen 启动进程
    process = subprocess.Popen(
        command1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    process.wait()


    # 将命令构建为一个列表（更安全，无需 shell=True）
    command = [
        "python", "/data/wlf/RAE-IGD-DMVAE/IGD/train.py",
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

    print("开始训练...")

    # 使用 Popen 启动进程
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='')

    # 等待进程结束并获取返回码
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