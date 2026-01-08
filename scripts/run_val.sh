cd /data/wlf/RAE-IGD-DMVAE

echo "脚本名称: $0"

python IGD/train.py -d imagenet --imagenet_dir $1 /data/wlf/datasets/imagenette/ -n resnet --depth 18 --nclass 10 --norm_type instance --ipc 10 --tag test --slct_type random --spec nette --batch_size 32 --verbose