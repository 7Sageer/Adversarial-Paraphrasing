from __future__ import annotations

import argparse

from datasets import load_from_disk
from tqdm import tqdm

import utils
from runtime_utils import LOCAL_DEPLOY_CLASSIFIER
from text_loader import load_initial_ai_text, load_wm_initial_text


def main(args):
    if args.deploy_classifier == "kgw_wm":
        original_texts = load_wm_initial_text("kgw")
    elif args.deploy_classifier == "uni_wm":
        original_texts = load_wm_initial_text("uni")
    else:
        original_texts = load_initial_ai_text(num_samples=args.num_samples)

    paraphrased_texts = load_from_disk(args.paraphrased_texts_path)["text"]
    if len(original_texts) < len(paraphrased_texts):
        raise RuntimeError("Original text count is smaller than paraphrased text count.")

    print("Loading deploy classifier...")
    deploy_classifier = utils.build_deploy_classifier(
        classifier_name=args.deploy_classifier,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
        watermark_tokenizer=args.watermark_tokenizer,
    )

    for i in tqdm(range(0, len(paraphrased_texts), args.batch_size)):
        batch_texts = original_texts[i : i + args.batch_size]
        outputs = paraphrased_texts[i : i + args.batch_size]

        if args.deploy_classifier in {"kgw_wm", "uni_wm"}:
            inp_scores, out_scores = [], []
            for input_text, output_text in zip(batch_texts, outputs):
                in_score_dict = deploy_classifier.detect(input_text)
                out_score_dict = deploy_classifier.detect(output_text)
                inp_scores.append(in_score_dict["z_score"])
                out_scores.append(out_score_dict["z_score"])
        elif args.deploy_classifier in {"fastdetectgpt", "gltr"}:
            inp_scores = deploy_classifier.inference(batch_texts)
            out_scores = deploy_classifier.inference(outputs)
        else:
            inp_scores = deploy_classifier.get_scores(batch_texts)
            out_scores = deploy_classifier.get_scores(outputs)

        for input_text, output_text, input_score, output_score in zip(
            batch_texts, outputs, inp_scores, out_scores
        ):
            print(
                f"\n\n\n{'*' * 20}\n<input>{input_text}</input>\n<inp_score>{input_score}</inp_score>\n\n"
                f"<output>{output_text}</output>\n<out_score>{output_score}</out_score>",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_cache_dir", type=str, default="")
    parser.add_argument("--watermark_tokenizer", type=str, default="")
    parser.add_argument("--paraphrased_texts_path", type=str, default="outputs/guided_generations_mage/adv/radar")
    parser.add_argument(
        "--deploy_classifier",
        type=str,
        choices=utils.DEPLOY_CLASSIFIER_CHOICES,
        default=LOCAL_DEPLOY_CLASSIFIER,
    )
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()
    print("*" * 20, "\n", args, "\n", "*" * 20, "\n")
    main(args)
