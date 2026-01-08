import subprocess
import argparse


def main(args):
    try:
        results = subprocess.run(['scripts/run_val.sh', args.save_dir], capture_output=True, text=True)
        print("STDOUT:")
        print(results.stdout)
        print("STDERR:")
        print(results.stderr
    except subprocess.CalledProcessError as e:
        print(e)
if __name__ ==  'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', default='/data2/wlf/fractalgen-distillation/nette', type=float)

    args = parser.parse_args()

    main(args)