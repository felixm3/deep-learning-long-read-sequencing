#!/bin/bash
#
#SBATCH --job-name=python_MULTIgpu_submit_ntasks_1_cpus_28_mem_120_231205
#SBATCH --output=/home/fmbuga/tools/slurm_scripts/slurm_out_err/python_MULTIgpu_submit_ntasks_1_cpus_28_mem_120_231205_%j.out
#SBATCH --error=/home/fmbuga/tools/slurm_scripts/slurm_out_err/python_MULTIgpu_submit_ntasks_1_cpus_28_mem_120_231205_%j.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --partition=gpus
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu01

datenow=$(date)
echo $datenow
echo ""
srun hostname
echo ""

start=$(date +%s)
echo "start time: $start"
echo ""
echo "hostname: $HOSTNAME"
echo ""

# for GPU activation
module load gnu/9.3.0 cuda-11.4
nvidia-smi
echo ""

##################

python_script=$1

killall R                    ### in case zombie R processes taking up RAM

eval "$(conda shell.bash hook)"
conda activate tf-gpu                      ### CONDA ENVIRONMENT NAME
echo ""

conda info
echo ""
conda list
echo ""

# GPU activated?
nvidia-smi
echo ""

python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
echo ""

work_dir=$(pwd)
echo "working directory: $work_dir"     ### WORKING DIRECTORY
echo ""


# python

call="python $python_script"                                   ### PYTHON SCRIPT TO SLURM SUBMIT
echo $call
eval $call
echo "python slurm submit COMPLETE."
echo ""
    

##################

echo ""
end=$(date +%s)
echo "end time: $end"
runtime_s=$(echo $(( end - start )))
echo "total run time(s): $runtime_s"
sec_per_min=60
sec_per_hr=3600
runtime_m=$(echo "scale=2; $runtime_s / $sec_per_min;" | bc)
echo "total run time(m): $runtime_m"
runtime_h=$(echo "scale=2; $runtime_s / $sec_per_hr;" | bc)
echo "total run time(h): $runtime_h"


