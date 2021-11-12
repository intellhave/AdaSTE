for i in 100
do
    python main.py --dataset tinyimg --model RESNET18 --optim FenBP --experiment-id $i --seed $i --val-split 0 --data_path /home/intellhave/Work/Datasets/TINYIMAGENET200/ --save-model --init_beta 0.01 --beta_inc_rate 1.025 --lr 1e-6 --batch-size 64 --log-interval 1500 --lr_schedular Exp
    #--resume_path /home/intellhave/Work/NetQuantz/BayesNN/pretrained/cifar10_resnet18.ckpt
    #python main_cifar10.py --model RESNET18 --optim BayesBiNN --experiment-id $i --seed $i
done
