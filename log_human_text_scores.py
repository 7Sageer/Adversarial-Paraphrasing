from __future__ import annotations

import argparse

from tqdm import tqdm

import utils
from runtime_utils import LOCAL_DEPLOY_CLASSIFIER, save_jsonl
from text_loader import load_initial_human_text


def main(args):
    human_texts = load_initial_human_text(num_samples=args.num_samples)
    print("Loading deploy classifier...")
    deploy_classifier = utils.build_deploy_classifier(
        classifier_name=args.deploy_classifier,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
        watermark_tokenizer=args.watermark_tokenizer,
    )

    results = []
    for i in tqdm(range(0, len(human_texts), args.batch_size)):
        batch_texts = human_texts[i : i + args.batch_size]

        if args.deploy_classifier in {"kgw_wm", "uni_wm"}:
            scores = []
            for text in batch_texts:
                score_dict = deploy_classifier.detect(text)
                scores.append(score_dict["z_score"])
        elif args.deploy_classifier in {"fastdetectgpt", "gltr"}:
            scores = deploy_classifier.inference(batch_texts)
        else:
            scores = deploy_classifier.get_scores(batch_texts)

        for idx, (text, score) in enumerate(zip(batch_texts, scores), start=i):
            results.append({"id": idx, "text": text, "score": float(score)})

    output_path = f"outputs/human_text_scores/data-mage_model-{args.deploy_classifier}.jsonl"
    save_jsonl(output_path, results)
    print(f"Saved {len(results)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_cache_dir", type=str, default="")
    parser.add_argument("--watermark_tokenizer", type=str, default="")
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
