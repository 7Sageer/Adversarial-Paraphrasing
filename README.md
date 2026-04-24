# Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text

This repository contains the official implementation of **Adversarial Paraphrasing**, introduced in the paper:  
📄 [_Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text_](https://www.arxiv.org/abs/2506.07001)

Adversarial Paraphrasing is a universal, training-free attack that humanizes AI-generated text to evade a wide variety of AI text detectors.



## 🧠 Overview
Adversarial paraphrasing is a training-free attack framework that universally humanizes any AI-generated text to evade detection more effectively. Our approach leverages an off-the-shelf instruction-following LLM to paraphrase AI-generated content under the guidance of an AI text detector, producing adversarial examples that are specifically optimized to bypass detection.

![Adversarial Paraphrasing Algorithm](readme_imgs/algorithm.png)

> **Key Insight**: Most, if not all, high-performing detectors tend to converge toward a common distribution that characterizes human-authored text. Consequently, if a paraphraser is guided to evade detection by a well-trained detector, its outputs will naturally align more closely with this shared, human-like distribution. As a result, the generated text becomes more difficult to detect—not only by the detector used for guidance, but also by others, since all are ideally calibrated to the same underlying distribution of human-written text.

Ours experiments show that our attack is both broadly effective and highly transferable across several detection systems.

![Main Results](readme_imgs/main_res.png)

For questions or feedback, please email [Yize Cheng](mailto:yzcheng@cs.umd.edu) and [Vinu Sankar Sadasivan](mailto:vinu@cs.umd.edu).


## 📁 Repository Structure
- `utils.py`: Core implementation of the Adversarial Paraphrasing algorithm.

- `quality_judge_utils.py`: Prompt templates for GPT-4o quality evaluation.

- `outputs/`: Saved outputs of simple and adversarially paraphrased texts, as well as detection scores on human texts that were used for plotting the ROC curve.

- `scripts/`: Scripts for running paraphrasing, detection, and dataset creation.

- `kgw_wm/wm_mage`, `uni_wm/wm_mage`: Watermarked datasets (KGW and Unigram).

- Implementation of detectors used in the paper.

- `parseNsave_paraphrased_output.py`: Parses and saves paraphrased outputs into HuggingFace dataset format.

## ⚙️ Installation
We recommend creating a separate virtual or conda environment with python>=3.10, and then run:
```bash
pip install -r requirements.txt
```

For local Apple Silicon testing, the full server snapshot in `requirements.txt` is heavier than necessary. The minimal local path is:

```bash
pip install torch transformers datasets tqdm clean-text sentencepiece scipy
```

## 💻 Local Apple Silicon Notes

The original codebase assumes a CUDA machine and large local models. This repository now includes a simplified local path that keeps the same script structure but is friendlier to Apple Silicon:

- `--device auto` resolves to `cuda`, `mps`, or `cpu`
- the default paraphraser is now `Qwen/Qwen2.5-0.5B-Instruct`
- the default local detector is now `openai_roberta_base`
- watermark detection still requires a compatible tokenizer path and is not the recommended first-run target on a laptop

The fastest smoke test is a single local paraphrase:

```bash
python paraphrase_and_detect.py \
  --input_text "Large language models can generate fluent text quickly, but detectors often look for statistical regularities." \
  --device auto \
  --adversarial 0
```

If you want lightweight detector-guided paraphrasing on a laptop, start with:

```bash
python paraphrase_and_detect.py \
  --input_text "Large language models can generate fluent text quickly, but detectors often look for statistical regularities." \
  --device auto \
  --guidance_classifier openai_roberta_base \
  --deploy_classifier openai_roberta_base \
  --adversarial 1
```

This local mode is for experimentation and debugging. It does not claim to reproduce the paper's full CUDA-scale setup on a MacBook.

## ☁️ Colab

A minimal Colab notebook is available at `notebooks/colab_minimal.ipynb`.

- It auto-selects a safe runtime profile based on detected GPU memory
- It is designed for single-text `text-in / text-out` runs
- It uses conservative defaults for T4 / L4 / A100-class runtimes

The notebook expects the patched Colab-friendly version of this repository. If you want to open it from GitHub in Colab, push this branch to your own fork first and update `REPO_URL` / `REPO_REF` in the notebook.


## 📦 Watermarked Datasets
We covered two types of watermarks in our experiments:
- **[KGW](https://arxiv.org/abs/2301.10226)**
- **[Unigram](https://arxiv.org/abs/2306.17439)**

The watermarked datasets are available under `kgw_wm/wm_mage/` and `uni_wm/wm_mage/` respectively (in HuggingFace format).

To recreate them:

```bash
# SLURM
sbatch scripts/create_wm_mage.sbatch

# or local CUDA
bash scripts/create_wm_mage.sbatch
```

> Customize the `wm_name` variable in the script to either `kgw_wm` or `uni_wm`.

## 🚀 Running Adversarial Paraphrasing

To run adversarial paraphrasing on original AI texts:
```bash
# SLURM
sbatch scripts/transfer_test.sbatch

# or local CUDA
bash scripts/transfer_test.sbatch > path_to_log_file.log
```
**Parameters to customize:**
- `dataset`: Input dataset (must match detector type, e.g. watermarked).
- `guidance_classifier`: Detector used to guide paraphrasing.
- `deploy_classifier`: Detector used to evaluate paraphrased outputs.
- `adversarial=1`: Enables adversarial paraphrasing.  
  Set to `0` for regular paraphrasing (guidance classifier is ignored in this case).

> When run with SLURM, outputs are saved in SLURM logs. In local environments, redirect output to a file to capture results.

The results include both the input and output texts, along with their detection scores. Lower scores indicate more 'human-like' content, while higher scores suggest more 'AI-generated' content.



## 🔁 Testing Other Detectors on Existing Outputs

Once you've generated adversarially paraphrased outputs, you can evaluate them using different detectors without re-paraphrasing:

1. **Extract the paraphrased outputs:**

   Edit and run:
   ```bash
   python parseNsave_paraphrased_output.py
   ```

2. **Run additional detectors on these outputs:**
   ```bash
   # SLURM
   sbatch scripts/detect_existing_paraphrased_text.sbatch

   # or local CUDA
   bash scripts/detect_existing_paraphrased_text.sbatch
   ```


## 🌟 Text Quality Evaluation (with GPT-4o)

We provide utilities to evaluate the quality of the paraphrased texts using GPT-4o.

1. Set your `OPENAI_API_KEY` as an environment variable.
2. Load saved texts in any way you prefer and pass them to the functions in `quality_judge_utils.py`.
3. The included prompt templates follow our experimental design for consistent quality evaluation.

## citation

If you find our work useful, please consider citing us via:

```bibtex
@misc{cheng2025adversarialparaphrasinguniversalattack,
      title={Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text}, 
      author={Yize Cheng and Vinu Sankar Sadasivan and Mehrdad Saberi and Shoumik Saha and Soheil Feizi},
      year={2025},
      eprint={2506.07001},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.07001}, 
}
```
