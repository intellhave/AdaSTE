LR=0.01
DEPTH=18
python3 main_fenbp.py  --model resnet --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH"_FenBP --dataset cifar10 --batch-size 64 --gpu 0 --epochs 600 --tb_dir tb/CIFAR10_FenBP --optimizer Adam --lr $LR --print-freq 390 --projection_mode lazy 

