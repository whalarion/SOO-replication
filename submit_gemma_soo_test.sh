#!/bin/bash
#SBATCH --job-name=gemma-soo-TEST   # Indicate this is a test job
#SBATCH --account=project_2013894  # Replace with your CSC project account ID
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --time=00:20:00             # <<< Short time for testing (e.g., 20 minutes)

# --- Load Required Modules ---
module purge
module load pytorch/1.13 # Or your preferred compatible version

# --- Activate Python Environment (Optional) ---
# source /path/to/your/conda/env/bin/activate
# source /path/to/your/venv/bin/activate

# --- Environment Variables ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=$LOCAL_SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
mkdir -p $TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1

# Define paths
TRAINING_SCRIPT="finetuning_test.py"
DS_CONFIG="ds_config_stage3.json"
TEST_DATA_FILE="soo_test_data10.jsonl" # <<< Point to the small test dataset
TEST_OUTPUT_DIR="/scratch/project_2013894/gemma-lora-soo-TEST-OUTPUT" # <<< Use a separate output dir for test

# Create test output directory
mkdir -p $TEST_OUTPUT_DIR

# --- Run Minimal Test with DeepSpeed ---
echo "Starting DeepSpeed TEST RUN on node $SLURMD_NODENAME"
echo "Time: $(date)"
echo "Using Test Dataset: $TEST_DATA_FILE"
echo "Outputting to: $TEST_OUTPUT_DIR"

# Launch with deepspeed, passing test-specific arguments to the Python script
srun deepspeed --num_gpus=$SLURM_GPUS_PER_TASK $TRAINING_SCRIPT \
    --deepspeed $DS_CONFIG \
    --data_file $TEST_DATA_FILE \
    --output_dir $TEST_OUTPUT_DIR \
    --model_name_or_path "google/gemma-2-27b-it" `# Or path if cached/copied` \
    --max_steps 10 `# <<< Limit to very few steps` \
    --no_save `# <<< Disable saving checkpoints` \

EXIT_CODE=$?

echo "Time: $(date)"
echo "DeepSpeed TEST RUN finished with exit code $EXIT_CODE."
echo "Check logs in $TEST_OUTPUT_DIR and Slurm output files."

exit $EXIT_CODE