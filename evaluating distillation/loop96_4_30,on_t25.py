import os

files = ["s"+str(i)+"_model.pth" for i in range(30,60)]


for file in files:
    os.system(f"python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --n_epochs 1 --reweight_groups --resnet_width 32 --show_progress --resume --log_every 1 --model_test {file} --log_dir ../temp2.96.4 --gpu 7")

    
for file in files:
    os.system(f"python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --n_epochs 1 --reweight_groups --resnet_width 32 --show_progress --resume --log_every 1 --model_test {file} --log_dir ../temp5.96.4 --gpu 7")