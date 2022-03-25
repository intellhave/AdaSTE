for i in 100
do
    python main_cifar100.py --model VGG16 --optim FenBP --experiment-id $i --seed $i
done
