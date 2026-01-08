import subprocess
import argparse


def main(args):
    subprocess.run(['scripts/run_val.sh', args.save_dir], capture_output=True, text=True)  # arg1, arg2是传给脚本的参数

if __name__ ==  'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', default='/data2/wlf/fractalgen-distillation/nette', type=float)

    args = parser.parse_args()

    main(args)