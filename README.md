This is the code to reproduce the results of "Model Distillation for Worst group generalization"


## Prerequisites
- python 3.6.8
- matplotlib 3.0.3
- numpy 1.16.2
- pandas 0.24.2
- pillow 5.4.1
- pytorch 1.1.0
- pytorch_transformers 1.2.0
- torchvision 0.5.0a0+19315e3
- tqdm 4.32.2

## Datasets and code 

To run our code, you will need to change the `root_dir` variable in `data/data.py`, in both distillation and evaluation directories
Below, we provide sample commands for each dataset.

### CelebA
Our code expects the following files/folders in the `[root_dir]/celebA` directory:

- `data/list_eval_partition.csv`
- `data/list_attr_celeba.csv`
- `data/img_align_celeba/`

You can download these dataset files from [this Kaggle link](https://www.kaggle.com/jessicali9530/celeba-dataset). The original dataset, due to Liu et al. (2015), can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).


To train a teacher model, you need to run the Sagawa code available at https://github.com/ssagawa/overparam_spur_corr.git

Once you have a trained teacher model, you need to specify its path in the teacher_dir flag of the command.

Note: a student model is saved for every epoch of distillation

Example command to distill would be:

cd distillation

nohup python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 60 --reweight_groups --train_from_scratch --resnet_width 32 --show_progress --log_every 1 --log_dir ../temp5.96.8 --teacher_dir ../96_best --teacher_width 96 --student_width 8 --temp 5 --gpu 5

Because this code is a modification of the sagawa code, there are some redundant flags which don't matter but need to be there to make sure the code runs. The flags that matter are:
 --n_epochs 60 --reweight_groups --train_from_scratch --resnet_width 32 --show_progress --log_every 1 --log_dir ../temp5.96.8 --teacher_dir ../96_best --teacher_width 96 --student_width 8 --temp 5

temp : specifies the temperature for distillation
gpu : To specify the gpu number on which you want do the distillation
student_width : width of the student model
teacher_width : width of the teacher
teacher_ : directory where the trained teacher model file is present
log_dir: directory where you want the log files and distilled models to saved.



To test a particular distilled model, example command would be:

cd evaluation
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --n_epochs 1 --reweight_groups --resnet_width 32 --show_progress --resume --log_every 1 --model_test last_model.pth --log_dir ../8 --gpu 2

Here you should be concerned about only the following flags:
gpu : To specify the gpu number on which you want do the evaluation
log_dir : path to the directory where the distilled student model is present
model_test: name of the student model file present in the log directory that you want to test



The results will be saved in a file ending with .pth.text

