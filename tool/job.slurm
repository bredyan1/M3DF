#!/bin/bash

#SBATCH --job-name=myFirstGPUJob    # 作业在调度系统中的作业名为myFirstJob
#SBATCH --partition=normal_test     # 作业提交的指定队列/分区为normal_test
#SBATCH --nodes=1                   # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=1         # 每节点任务数，GPU任务不需要修改
#SBATCH --cpus-per-task=3           # V100一张卡默认配置3个CPU核心，gpuB一张卡默认配置12个CPU核心,MIG资源一张卡默认配置6个CPU核心(根据卡数自行调整)
#SBATCH --gres=gpu:1                # 申请一块GPU卡
#SBATCH -o %J.out                   # 脚本执行的输出将被保存在当 %J.out文件下，%j表示作业号
#SBATCH -e %J.err                   # 脚本执行的错误日志将被保存在当 %J.err文件下，%j表示作业号
#SBATCH --mail-type=BEGIN,END,FAIL  # 任务开始，结束，失败时邮件通知
#SBATCH --mail-user=2083663502@qq.com # 邮件通知邮箱
module load anaconda3               # 加载相关依赖
source activate mcat                # 如果已经在命令行中激活对应环境，提交脚本时需注释此行，推荐保留此行在base环境下提交任务
python main_rpg.py  --which_splits 5fold --split_dir tcga_gbmlgg --mode rpgattn --model_type rpgcat --fusion concat --apply_sig --lr 1e-4 --gc 32 --max_epochs 20                       # 执行相关命令