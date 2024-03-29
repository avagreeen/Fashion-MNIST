#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short

# The default run (wall-clock) time is 1 minute
#SBATCH --time=04:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# The default number of CPUs per task is 1, however CPUs are always allocated per 2, so for a single task you should use 2
#SBATCH --cpus-per-task=2

# The default memory per node is 1024 megabytes (1GB)
#SBATCH --mem=40960

#SBATCH --gres=gpu:1

# Set mail type to 'END' to receive a mail when the job finishes (with usage statistics)
#SBATCH --mail-type=END

# Your job commands go below here

# Uncomment these lines when your job requires this software
module use /opt/insy/modulefiles
module load cuda/9.0 cudnn/9.0-7.1.4

# Complex or heavy commands should be started with 'srun' (see 'man srun' for more information)
# (This is just an example, srun is of course not necessary for this command.)
srun --output='model1.out' python run_model1.py --epoch=100 --model=mynet4 --optim=adam --lr=0.0001  --sch=True --loss=smooth --alpha=0.1
