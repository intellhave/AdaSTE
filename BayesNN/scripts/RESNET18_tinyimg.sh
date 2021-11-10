for i in 100
do
    python main_tinyimg.py --model RESNET18 --optim BayesBiNN --experiment-id $i --seed $i --batch-size 128 --lr 3e-4 --beta_inc_rate 1.009
    #python main_cifar10.py --model RESNET18 --optim BayesBiNN --experiment-id $i --seed $i
done
# for i in 100 200 300 400 500
# do
#     python main_cifar10.py --model RESNET18 --optim BayesBiNN --experiment-id $i --seed $i
# done
