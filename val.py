import os
import subprocess
import argparse
import sys
import re  # 用于正则提取
import matplotlib.pyplot as plt  # 用于绘图


def parse_accuracy(line):
    """
    根据你的 train.py 输出日志格式提取准确率。
    假设输出包含类似 "Test Acc: 85.4" 或 "Accuracy: 85.4" 的字样。
    你需要根据实际控制台打印的内容修改正则表达式。
    """
    # 示例正则：匹配 "Acc: 12.34" 或 "Accuracy: 12.34"
    # match = re.search(r'(?:Acc|Accuracy)[:\s]+(\d+\.\d+)', line, re.IGNORECASE)

    # 另一种常见情况： "Test set: Average loss: ..., Accuracy: 85/100 (85.00%)"
    match = re.search(r'Accuracy:.*?(\d+\.\d+)%', line)

    # 如果你的输出只是简单的数字，或者格式不同，请调整这里
    if match:
        return float(match.group(1))
    return None


def main(args):
    work_dir = "/data/wlf/RAE-IGD-DMVAE"
    steps = os.listdir(args.save_dir)
    results = []

    for s in range(1, len(steps)):
        print(f'\n{"=" * 20}\n开始评估数据集 Step: {s}\n{"=" * 20}')

        # 构造路径
        distilled_data_path = os.path.join(args.save_dir, str(s))

        # 2. 构建命令
        # 注意：--imagenet_dir 后面紧跟两个独立的元素
        command = [
            "python", "IGD/train.py",
            "-d", "imagenet",
            "--imagenet_dir", distilled_data_path, args.imagenet_path,
            "-n", "resnet",
            "--depth", "18",
            "--nclass", "10",
            "--norm_type", "instance",
            "--ipc", "10",
            "--tag", "test",
            "--slct_type", "random",
            "--spec", "nette",
            "--batch_size", "32",
            "--epochs", f"{(s+1)*100}",
            "--verbose"
        ]

        print(f"执行命令: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        final_acc = 0.0

        # 3. 实时读取输出并抓取准确率
        for line in process.stdout:
            print(line, end='')  # 实时打印到屏幕，方便你看进度

            # 尝试提取准确率，通常取最后一次出现的准确率作为最终结果
            acc = parse_accuracy(line)
            if acc is not None:
                final_acc = acc

        return_code = process.wait()

        if return_code == 0:
            print(f"\nStep {s} 训练完成，提取到的准确率: {final_acc}%")
            # 尝试将 step 转为 int 方便画图
            try:
                step_val = int(s)
            except:
                step_val = s
            results.append((step_val, final_acc))
        else:
            print(f"\nStep {s} 训练出错，返回码: {return_code}")

    # 4. 绘图
    print("\n所有任务完成，开始绘图...")
    if len(results) > 0:
        # 解压数据
        x_steps, y_accs = zip(*results)

        plt.figure(figsize=(10, 6))
        plt.plot(x_steps, y_accs, marker='o', linestyle='-', color='b')
        plt.title('Step vs Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)

        # 保存图片
        output_img = 'step_accuracy_plot.png'
        plt.savefig(output_img)
        print(f"折线图已保存至: {os.path.abspath(output_img)}")
        # plt.show() # 如果是在服务器无界面环境，不要调用 show()
    else:
        print("没有获取到有效的准确率数据，无法绘图。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 注意路径末尾不要带空格
    parser.add_argument('--save_dir', default='/data/wlf/fractalgen-distillation/nette', type=str)
    parser.add_argument('--imagenet_path', default="/data/wlf/datasets/imagenette/", type=str)

    args = parser.parse_args()

    main(args)