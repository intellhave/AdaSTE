for i in 100 200 300 400 500
do
    python main_cifar100.py --model VGG16 --optim BayesBiNN --experiment-id $i --seed $i
done