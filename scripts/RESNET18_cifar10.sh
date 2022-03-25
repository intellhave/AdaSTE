for i in 100 200 300 400 500
do
    python main_cifar10.py --model RESNET18 --optim FenBP --experiment-id $i --seed $i
done
