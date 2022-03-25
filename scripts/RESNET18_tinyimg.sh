for i in 100 200 300 400 500
do
    python main_tinyimg.py --model RESNET18 --optim FenBP --experiment-id $i --seed $i --batch-size 128 --lr 3e-4 --beta_inc_rate 1.009
done
