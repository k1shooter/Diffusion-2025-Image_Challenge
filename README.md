# Image Generation Challenge



**Mid Evaluation Submission Due:** November 2 (Sunday) 23:59 KST  
**Final Submission Due:** November 15 (Saturday) 23:59 KST  
**Where to Submit:** KLMS

## Overview
<img width="384" height="192" alt="Image" src="https://github.com/user-attachments/assets/13225cab-ccfe-40da-badf-e402243721e4" />

In this challenge, you will train an image diffusion/flow model beyond the previous 2D toy experiment setups from the assignments. After training, you are encouraged to explore and apply any techniques you find effective for achieving high-quality generation with only a few sampling steps.

**Dataset:** Simpsons Face images (automatically downloaded by the provided script)

**Evaluation:** FID (Fréchet Inception Distance) scores at **NFE=1, 2, and 4** (Number of Function Evaluations)

## Environment Setup

```shell
git clone https://github.com/KAIST-Visual-AI-Group/Diffusion-2025-Image_Challenge
cd Diffusion-2025-Image_Challenge
conda create -n image_gen python=3.10 -y
conda activate image_gen
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Note:** `requirements_full.txt` contains the complete list of all packages installed in the TA's evaluation environment. You can refer to this file to check the exact versions of all libraries that will be available during evaluation.

## Challenge Structure

```
image_generation_challenge/
├── src/
│   ├── base_model.py           # Abstract base classes (provided, DO NOT MODIFY)
│   ├── network.py              # U-Net architecture (provided, DO NOT MODIFY)
│   │                           # Fixed hyperparameters: ch=128, ch_mult=[1,2,2,2], 
│   │                           # attn=[1], num_res_blocks=4, dropout=0.1
│   ├── module.py               # U-Net building blocks (provided, DO NOT MODIFY)
│   └── utils.py                # Utility functions (provided, CAN modify)
├── data/
│   ├── train_split.txt         # Train/val split file (provided, DO NOT MODIFY)
│   └── val_split.txt           # Train/val split file (provided, DO NOT MODIFY)
├── custom_model.py             # Template for implementation (students SHOULD modify)
├── train.py                    # Training script (students CAN modify)
├── dataset.py                  # Dataset loading (provided, DO NOT MODIFY)
├── sampling.py                 # Sampling script (provided, DO NOT MODIFY)
└── measure_fid.py              # FID evaluation (provided, DO NOT MODIFY)
```

**Legend:**
- **DO NOT MODIFY**: Keep these files as-is (for fair comparison)
- **SHOULD modify**: Main files where you implement your solution
- **CAN modify**: Optional modifications to improve your model

## What to Do

### Task 1: Implementing a Diffusion/Flow Wrapper

We provide the architecture backbone (which should remain fixed), but everything around it is up to you. Your goal is to design and implement your own diffusion/flow model wrapper, including:

- **Noise schedulers**: Control the noise schedule during training and sampling
- **Forward process**: Transform clean data to noisy data
- **Reverse process**: Denoise and generate samples from noise

Implement the classes in `custom_model.py` by inheriting from the base classes:

- **`CustomScheduler`**: Inherit from `BaseScheduler` and implement methods for:
  - `sample_timesteps(batch_size, device)`: Sample random timesteps for training
  - `forward_process(data, noise, t)`: Apply forward process to add noise to clean data
  - `reverse_process_step(xt, pred, t, t_next)`: Perform one denoising step
  - `get_target(data, noise, t)`: Get the target for model prediction

- **`CustomGenerativeModel`**: Inherit from `BaseGenerativeModel` and implement methods for:
  - `compute_loss(data, noise, **kwargs)`: Compute the training loss
  - `predict(xt, t, **kwargs)`: Make prediction given noisy data and timestep
  - `sample(shape, num_inference_timesteps=20, **kwargs)`: Generate samples from noise

You are free to add additional functions as needed for your implementation.

**Note on Additional Conditioning:**
The provided U-Net supports an optional `use_additional_condition` flag. When enabled, the network can accept an additional scalar condition (e.g., step size in Shortcut Models or end timestep `s` in Consistency Trajectory Models). This is useful for advanced few-step generation techniques that require conditioning on additional timestep-like information beyond the main diffusion timestep.

**Training Your Model:**

```bash
python train.py --num_iterations 100000 --batch_size 32 --device cuda
```

You can modify `train.py` to add custom training logic (learning rate schedules, optimizers, EMA, etc.)
### Task 2: Improving Few-Step Generation

Once your diffusion/flow wrapper is ready, the main challenge is to investigate and improve the generation quality with very few sampling steps.

- Your models will be evaluated with **NFE=1, 2, and 4**
- You are encouraged to experiment with techniques such as **Consistency Models**, **ReFlow**, or any other advanced methods you find effective
- Check out the Recommended Readings section, but you are not limited to implementing one of the algorithms introduced in those papers



## Important Rules

**PLEASE READ THE FOLLOWING CAREFULLY!** Any violation of the rules or failure to properly cite existing code, models, or papers used in the project in your write-up will result in a zero score.

### What You CANNOT Do

- ❌ **Do NOT use pre-trained diffusion models**: You must train the model from scratch
- ❌ **Do NOT modify the provided U-Net architecture code**: Network hyperparameters are FIXED
- ❌ **Do NOT modify the provided sampling code and evaluation script**: These will be distributed to ensure consistent evaluation across all submissions
- ❌ **Do NOT modify the provided train/val split files**: `data/train_split.txt` and `data/val_split.txt` are provided for consistent data splitting
- ❌ **Do NOT install additional libraries separately**: Your code will be run in the TA's environment with the provided dependencies only. If you believe a specific library is essential for your implementation and many students have the same need, please request it on Slack. If there is sufficient demand, it will be officially announced and added to the environment.

### What You CAN Do

- ✅ **Modify `custom_model.py`**: Implement your scheduler and model classes
- ✅ **Modify `train.py`**: Add custom training logic, optimizers, learning rate schedulers, custom arguments, etc.
  - **Note**: Any custom arguments you add to `train.py` (except training-specific ones like `--lr`, `--batch_size`) will be automatically saved to `model_config.json` and loaded during sampling
- ✅ **Modify `src/utils.py`**: Add utility functions as needed
- ✅ **Create new files**: Add any additional implementation files you need
- ✅ **Use open-source implementations**: As long as they are clearly mentioned and cited in your write-up


## Evaluation

The performance of your image generative models will be evaluated quantitatively using FID scores at NFEs = 1, 2, and 4.

- Final grading will be determined relative to the best FID score achieved at each NFE
- The team with the lowest (best) FID for a given NFE will set the benchmark
- You are expected to match or surpass the TA's baseline FID scores

**TA's Baseline (Rectified Flow):**

| NFE | FID Score |
|-----|-----------|
| 1   | 293.36    |
| 2   | 233.82    |
| 4   | 114.27    |

## Submissions

### Mid-Term Evaluation (Optional)

**Due:** November 2 (Sunday) 23:59 KST  
**Where:** KLMS

The purpose of the mid-term evaluation is to give all students a reference point for how other teams are progressing. **Participation is optional**, but the top team at each NFE that also outperforms the TAs' FID scores will receive bonus credit toward the final grade.

**What to Submit:**

1. **Self-contained source code**
   - Complete codebase that can run end-to-end from the TAs' side
   - TAs will run your code in their environment without modifications
   - **Note:** `sampling.py` and `measure_fid.py` will be replaced with official versions for consistent evaluation

2. **Model checkpoint and config JSON file**
   - Save your best checkpoint as `./checkpoints/best_model.pt`
   - Include `./checkpoints/model_config.json` (auto-generated during training)

**Submission Structure:**
```
your_submission/
├── src/
├── custom_model.py (or your implementation files)
├── train.py
├── dataset.py
├── requirements.txt
├── checkpoints/
│   ├── best_model.pt         # ← REQUIRED
│   └── model_config.json     # ← REQUIRED
└── results/                   # Optional
```

**Evaluation:**
- TAs will run your submitted code and measure FID scores at NFE=1, 2, and 4
- Results will be published on the leaderboard
- Submissions that fail to run will be marked as **failed** on the leaderboard
- **Among submissions exceeding the TAs' result, the top-k will earn bonus credit**

### Final Submission

**Due:** November 15 (Saturday) 23:59 KST  
**Where:** KLMS

**What to Submit:**

1. **Self-contained source code** (same as mid-term)
2. **Model checkpoint and config JSON file** (same as mid-term)
3. **2-page write-up** (PDF format)

**Write-up Requirements:**
- Maximum **two A4 pages**, excluding references
- **Must include ALL of the following:**
  - **Technical details**: One-paragraph description of your few-step generation implementation
  - **Training details**: Training logs (e.g., loss curves) and total training time
  - **Qualitative evidence**: ~8 sample images from early training phases
  - **Citations**: All external code and papers used must be properly cited
- ⚠️ **Missing any of these items will result in a 10% penalty for each**
- ⚠️ **If the write-up exceeds two pages, any content beyond the second page will be ignored**, which may lead to missing required items

## Grading

- **Quantitative Evaluation**: FID scores at NFE=1, 2, and 4 (officially computed by TAs)
- **Leaderboard Performance**: Top performers receive bonus credit
- **Write-up**: Clear technical explanation and proper citations

### Important

- ⚠️ **There is no late day. Submit on time.**
- ⚠️ **Late submission: Zero score**
- ⚠️ **Missing any required item in the final submission (samples, code/model, write-up): Zero score**
- ⚠️ **Missing items in the write-up: 10% penalty for each**
- ⚠️ **Citation is mandatory**: Any violation of the rules or failure to properly cite existing code, models, or papers used in the project will result in a zero score

## Quick Start

### 1. Train Your Model

```bash
python train.py --num_iterations 100000 --batch_size 32 --device cuda
```

Monitor training progress in `./results/TIMESTAMP/training_curves.png`

### 2. Generate Samples

```bash
python sampling.py \
    --ckpt_path ./results/TIMESTAMP/last.ckpt \
    --save_dir ./samples
```

### 3. Evaluate FID (Optional - TAs will do official evaluation)

```bash
# Reference images will be automatically prepared on first run under the ./data/simpsons_64x64/val directory.
# Evaluate each NFE separately
python measure_fid.py --generated_dir ./samples/nfe=1
python measure_fid.py --generated_dir ./samples/nfe=2
python measure_fid.py --generated_dir ./samples/nfe=4
```

## Self-Evaluation Checklist

Before submitting, verify:

1. ✅ **Code runs end-to-end**: Train → Sample → Evaluate without errors
2. ✅ **Checkpoint compatibility**: Works with official `sampling.py` (will be replaced by TAs)
3. ✅ **NFE=1,2,4 tested**: Your model generates reasonable samples at these NFE values
4. ✅ **All required files included**: Source code, checkpoints, config JSON, write-up (final submission)
5. ✅ **Citations ready**: All external code/papers properly cited in write-up

## Recommended Readings (Few-Step Generation)

- [1] [**Consistency Models**](https://arxiv.org/abs/2303.01469) (Song et al., ICML 2023)
- [2] [**Shortcut Models**](https://arxiv.org/abs/2410.12557) (Frans et al., ICLR 2025)
- [3] [**Flow Straight and Fast: Rectified Flow**](https://arxiv.org/abs/2209.03003) (Liu et al., ICLR 2023)
- [4] [**Progressive Distillation for Fast Sampling**](https://arxiv.org/abs/2202.00512) (Salimans & Ho, ICLR 2022)
- [5] [**Learning to Discretize Denoising Diffusion ODEs**](https://arxiv.org/abs/2405.15506) (Tong et al., ICLR 2025)
- [6] [**Adversarial Diffusion Distillation**](https://arxiv.org/abs/2311.17042) (Sauer et al., ECCV 2024)

## Tips

1. **Start Simple**: Begin with basic Flow Matching or DDPM, then add optimizations
2. **Test Incrementally**: Verify each component before combining
3. **Monitor Training**: Check loss curves and sample quality regularly
4. **Focus on NFE=1,2,4**: Optimize specifically for few-step generation
5. **Citation**: Always cite external code and papers properly

---

Good luck! 🚀
