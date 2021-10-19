LR=0.01
DEPTH=50
dataset=cifar100
python3 main_fenbp.py  --model resnet --model_config "{'depth': $DEPTH}" --save resnet"$DEPTH"_"$dataset"_FenBP --dataset $dataset --batch-size 128 --gpu 0 --epochs 600 --tb_dir tb/CIFAR10_FenBP --optimizer Adam --lr $LR --print-freq 150 --binary_regime --freeze_epoch 200
#--delta_decrease_epoch 1 --init_delta 1e-6 --init_eta 0.15
#--resume /home/intellhave/Work/NetQuantz/results/resnet20_FenBP_D1/model_best.pth.tar
#--resume /home/intellhave/Work/NetQuantz/results/resnet20/
#--resume /home/intellhave/Work/NetQuantz/results/resnet20_FenBP_L3/
#--resume /home/intellhave/Work/NetQuantz/results/resnet20_FenBP_1em4_cosine_f4_sub2/
#--resume /home/intellhave/Work/NetQuantz/results/resnet20_FenBP_1em4_cosine_f4/




