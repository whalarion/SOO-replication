# Replication of Self-Other Overlap (SOO) Fine-tuning for Gemma-2-27B

*Status: Project is currently held up due to HPC resource availability and I'm still troubleshooting. Code will not work as is.*

## Project Overview

This is a work in progess project aiming to replicate the Self-Other Overlap (SOO) fine-tuning methodology presented in the paper "Towards Safe and Honest AI Agents with Neural Self-Other Overlap" (Carauleanu et al., 2024) focusing on its application to the Gemma-2-27B language model. I'm using Gemma-3-4B-it as I only have access to V100s and this is said to have similar general performance. 

As the original code and fine-tuning data are not publicly available, this repository contains custom implementations of the data generation process and the SOO fine-tuning training script, developed with assistance from large language models (about 50% of work done by Gemini 2.5 Pro and GPT-4o). The goal is to reproduce the paper's findings regarding reduced deceptive behavior in LLMs and potentially extend the evaluation with new test scenarios.

Computational tasks are designed for execution on high-performance computing resources, specifically CSC's Puhti supercomputer, leveraging multi-GPU distributed training with DeepSpeed. 

## Methodology

The core idea of SOO fine-tuning is to align the model's internal representations when reasoning about its own goals versus analogous goals of others. This is achieved by minimizing the difference (e.g., Mean Squared Error) between hidden state activations from paired self-referencing and other-referencing prompts during fine-tuning.

This repository implements the following workflow:

1.  **Data Generation:** `generate_data.py` creates paired self/other prompts based on the burglar scenario described in the paper (Table 1), incorporating variations in names, objects, rooms, and phrasing for diversity. The output is saved in JSON Lines (`.jsonl`) format.
2.  **Fine-tuning:** The primary Python training script (`finetuning.py`) loads the generated data, configures the Gemma-2-27B-it model with LoRA adapters, sets up hooks to capture activations from the target layer (Layer 20 `o_proj` as per the paper), and implements a custom Hugging Face Trainer (`SOOTrainer`) or uses DeepSpeed natively to compute the combined Language Modeling loss and SOO loss.
3.  **Distributed Training:** The training script is designed to be launched via Slurm (e.g., `submit_gemma_soo.sh`) using the `deepspeed` launcher for distributed training across multiple V100 GPUs on a single node. DeepSpeed configuration (e.g., `ds_config_stage3.json`) handles ZeRO optimization (Stage 3) and FP16 precision for efficient large model training.

## Repository Structure

*   `generate_data.py`: Script to generate the paired SOO fine-tuning data (`.jsonl` format).
*   `finetuning.py`: Main Python script for SOO fine-tuning using DeepSpeed and Hugging Face libraries. (Or your primary training script name)
*   `ds_config_stage3.json`: DeepSpeed configuration file for ZeRO Stage 3 optimization.
*   `submit_gemma_soo.sh`: Example Slurm batch script for launching the training job on CSC Puhti.
*   `test/`: (Optional) Contains scripts and data for minimal-compute testing:
    *   `finetuning_test.py`: A version of the training script configured for short test runs.
    *   `submit_gemma_soo_test.sh`: Example Slurm script for submitting a short test job.
    *   `soo_finetuning_data10.jsonl`: Small example dataset for testing.

## Usage workflow for the training

1.  **Setup:**
    *   Installing required Python packages (e.g., `torch`, `transformers`, `peft`, `datasets`, `deepspeed`, `accelerate`). A Python virtual environment is recommended.
    *   Accessing HPC environment.
2.  **Data Generation:**
    *   Running `python generate_data.py` to get the amount of examples wanted.
3.  **Configuration:**
    *   Modifying the Slurm script (`submit_gemma_soo.sh`) for correct credentials and resource requests.
    *   Adjusting the DeepSpeed configuration (`ds_config_stage3.json`) and `TrainingArguments` within the Python script.
    *   Ensuring file paths within the scripts are correct.
4.  **Testing:**
    *   Running the scrum script (`submit_gemma_soo_test.sh`) and test training script (`finetuning_test.py`), to verify the setup with minimal compute usage.
5.  **Training:**
    *   Submitting the main Slurm job: `sbatch submit_gemma_soo.sh`.
    *   Monitoring the job progress and output files.

## Citation

Carauleanu, M., Vaiana, M., Rosenblatt, J., Berg, C., & de Lucena, D. S. (2024). Towards Safe and Honest AI Agents with Neural Self-Other Overlap. *arXiv preprint arXiv:2412.16325*.

## Acknowledgements

Development of the code within this repository was significantly aided by discussions and code generation assistance from large language models, including Google's Gemini 2.5 Pro Experimental and OpenAI's GPT-4o. Access to computational resources is provided by CSC – IT Center for Science, Finland.
