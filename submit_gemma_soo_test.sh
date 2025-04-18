#!/bin/bash
#SBATCH --job-name=gemma-soo-TEST
#SBATCH --account=project_2013894
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100:4,nvme:80
#SBATCH --cpus-per-task=4
#SBATCH --mem=373G
#SBATCH --time=01:00:00

# --- Load Required Modules ---
module purge
module load pytorch

# --- Activate Python Environment ---
source /projappl/project_2013894/soo/soo_env/bin/activate

# --- Environment Variables ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # OpenMP for multi cpu jobs
export HF_HOME=$LOCAL_SCRATCH/.cache/huggingface # Where HF puts files
export HF_DATASETS_CACHE=$LOCAL_SCRATCH/.cache/huggingface/datasets # Where HF datasets library puts files
mkdir -p $HF_DATASETS_CACHE # Ensure it exists
mkdir -p $HF_HOME # Ensure base cache dir exists on NVMe
export PYTHONUNBUFFERED=1 # Disables buffering for easier debugging
export HF_TOKEN="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"  # Gemma is gated so this is needed to download

# Define paths
TRAINING_SCRIPT="finetuning_test.py"
DS_CONFIG="ds_config_stage3.json"
TEST_DATA_FILE="soo_test_data10.jsonl"
TEST_OUTPUT_DIR="/scratch/project_2013894/gemma-lora-soo-TEST-OUTPUT"

# Create test output directory
mkdir -p $TEST_OUTPUT_DIR

# --- Run Minimal Test with DeepSpeed ---
echo "Starting DeepSpeed TEST RUN on node $SLURMD_NODENAME"
echo "Time: $(date)"
echo "Using Test Dataset: $TEST_DATA_FILE"
echo "Outputting to: $TEST_OUTPUT_DIR"

# Launch with deepspeed
srun deepspeed $TRAINING_SCRIPT \
    --deepspeed $DS_CONFIG \
    --data_file $TEST_DATA_FILE \
    --output_dir $TEST_OUTPUT_DIR \
    --model_name_or_path "/scratch/project_2013894/models" \
    --max_steps 10 \
    --per_device_train_batch_size 1 \

EXIT_CODE=$? # Captures error code of "srun deepspeed"

echo "Time: $(date)"
echo "DeepSpeed TEST RUN finished with exit code $EXIT_CODE."
echo "Check logs in Slurm output file (slurm-<jobid>.out)."

exit $EXIT_CODE