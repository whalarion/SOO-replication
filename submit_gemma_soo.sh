#!/bin/bash
#SBATCH --job-name=gemma-soo-ds     # Job name for identification
#SBATCH --account=<your_project_account>  # Replace with your CSC project account ID
#SBATCH --partition=gpu             # Use the GPU partition
#SBATCH --nodes=1                   # Request one full node
#SBATCH --ntasks-per-node=4         # Run 4 tasks (one per GPU)
#SBATCH --gres=gpu:v100:4           # Request all 4 V100 GPUs on the node
#SBATCH --cpus-per-task=8           # Request 8 CPUs per GPU task (adjust 6-10 if needed)
#SBATCH --mem=350G                  # Request ample node memory (adjust if needed, but keep high)
#SBATCH --time=24:00:00             # Initial time request (e.g., 24 hours, adjust after first run)
#SBATCH --tmp=150G                  # Request 150GB local NVMe storage (adjust if model cache + data needs more/less)

# --- Load Required Modules ---
# This might vary slightly depending on the default environment or specific PyTorch versions available
module purge
module load pytorch/1.13 # Or a newer compatible PyTorch module available on Puhti
# Check if a specific deepspeed module is needed/available, otherwise rely on pip install
# module load deepspeed/0.7.0 # Example if available

# --- Activate Python Environment (Recommended) ---
# Replace with the path to your environment if you use Conda or venv
# source /path/to/your/conda/env/bin/activate
# source /path/to/your/venv/bin/activate
# If not using an env, ensure required packages (transformers, peft, datasets, deepspeed, accelerate) are installed in your default Python

# --- Environment Variables ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set Hugging Face cache and home to use the fast local NVMe drive
export HF_HOME=$LOCAL_SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
mkdir -p $TRANSFORMERS_CACHE # Ensure directory exists

# Optional: Set NCCL debug level if troubleshooting communication issues
# export NCCL_DEBUG=INFO

# Define paths (adjust if your script/output locations differ)
export PYTHONUNBUFFERED=1 # Ensure Python output is not buffered, good for logs
TRAINING_SCRIPT="train_soo_v100.py" # Your main Python training script
DS_CONFIG="ds_config_stage3.json" # Your DeepSpeed configuration file
OUTPUT_DIR="/scratch/<your_project>/gemma-lora-soo-finetuned-v100-ds" # Output on scratch

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# --- Run Training with DeepSpeed ---
echo "Starting DeepSpeed training on node $SLURMD_NODENAME"
echo "Time: $(date)"
echo "Using $SLURM_NTASKS tasks with $SLURM_GPUS_PER_TASK GPUs each."
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

# `srun` launches the parallel tasks.
# `deepspeed` command is executed by each task, coordinating via MPI/NCCL.
# Pass the path to your training script and the deepspeed config.
srun deepspeed --num_gpus=$SLURM_GPUS_PER_TASK $TRAINING_SCRIPT \
    --deepspeed $DS_CONFIG \
    # Add any other command line arguments needed by train_soo_v100.py AFTER the script name
    # Example: --output_dir $OUTPUT_DIR (if you made output_dir an argument in Python)

EXIT_CODE=$? # Capture exit code

echo "Time: $(date)"
echo "DeepSpeed training finished with exit code $EXIT_CODE."

# --- Optional: Post-job cleanup ---
# If needed, copy specific final files from $LOCAL_SCRATCH back to /scratch
# Example: cp $LOCAL_SCRATCH/some_intermediate_file.log $OUTPUT_DIR/

echo "Slurm job finished."

exit $EXIT_CODE