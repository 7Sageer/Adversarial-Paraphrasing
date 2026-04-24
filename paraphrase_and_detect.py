from __future__ import annotations

import argparse

import numpy as np
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

import utils
from runtime_utils import (
    LOCAL_DEPLOY_CLASSIFIER,
    LOCAL_GUIDANCE_CLASSIFIER,
    LOCAL_PARAPHRASER_MODEL,
)


def load_input_texts(args):
    np.random.seed(0)
    if args.input_text:
        return [args.input_text]

    if args.dataset == "mage":
        dataset = load_dataset("yaful/MAGE")["test"]
        labels = np.array([x["label"] for x in dataset])
        idx = np.arange(len(labels))[labels == 0]
        idx = np.random.choice(idx, args.num_samples * 10, replace=False)

        def condition(text):
            return args.n_words_sample <= len(text.split(" ")) <= 2 * args.n_words_sample

        texts = [dataset[int(i)]["text"] for i in idx if condition(dataset[int(i)]["text"])]
        return texts[: args.num_samples]

    if args.dataset == "kgwwm_mage":
        dataset = load_from_disk("kgw_wm/wm_mage")
        return dataset["text"]

    if args.dataset == "uniwm_mage":
        dataset = load_from_disk("uni_wm/wm_mage")
        return dataset["text"]

    raise ValueError(f"Unsupported dataset '{args.dataset}'.")


def get_output_scores(deploy_classifier_name, deploy_classifier, batch_texts, outputs):
    if deploy_classifier_name in {"kgw_wm", "uni_wm"}:
        inp_scores, out_scores = [], []
        for input_text, output_text in zip(batch_texts, outputs):
            in_score_dict = deploy_classifier.detect(input_text)
            out_score_dict = deploy_classifier.detect(output_text)
            inp_scores.append(in_score_dict["z_score"])
            out_scores.append(out_score_dict["z_score"])
        return inp_scores, out_scores

    if deploy_classifier_name in {"fastdetectgpt", "gltr"}:
        return deploy_classifier.inference(batch_texts), deploy_classifier.inference(outputs)

    return deploy_classifier.get_scores(batch_texts), deploy_classifier.get_scores(outputs)


def main(args):
    texts = load_input_texts(args)
    if not texts:
        raise RuntimeError("No input texts were loaded.")

    guidance_classifier = None
    if bool(args.adversarial):
        print("Loading guidance classifier...")
        guidance_classifier = utils.build_guidance_classifier(
            classifier_name=args.guidance_classifier,
            device=args.device,
        )

    print("Loading deploy classifier...")
    deploy_classifier = utils.build_deploy_classifier(
        classifier_name=args.deploy_classifier,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
        watermark_tokenizer=args.watermark_tokenizer,
    )

    paraphraser = utils.Paraphraser(
        name=args.model,
        classifier=guidance_classifier,
        device=args.device,
        precision=args.precision,
        top_k=args.top_k,
    )

    print(f"Loaded {len(texts)} texts.")
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch_texts = texts[i : i + args.batch_size]
        outputs = paraphraser.paraphrase(
            batch_texts,
            batch_size=len(batch_texts),
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            adversarial=args.adversarial,
            option=args.option,
            deterministic=bool(args.deterministic),
        )
        inp_scores, out_scores = get_output_scores(
            deploy_classifier_name=args.deploy_classifier,
            deploy_classifier=deploy_classifier,
            batch_texts=batch_texts,
            outputs=outputs,
        )

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
    parser.add_argument("--dataset", type=str, choices=["mage", "kgwwm_mage", "uniwm_mage"], default="mage")
    parser.add_argument("--input_text", type=str, default="")
    parser.add_argument("--model", type=str, default=LOCAL_PARAPHRASER_MODEL)
    parser.add_argument(
        "--guidance_classifier",
        type=str,
        choices=utils.GUIDANCE_CLASSIFIER_CHOICES,
        default=LOCAL_GUIDANCE_CLASSIFIER,
    )
    parser.add_argument(
        "--deploy_classifier",
        type=str,
        choices=utils.DEPLOY_CLASSIFIER_CHOICES,
        default=LOCAL_DEPLOY_CLASSIFIER,
    )
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--precision", type=str, choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--watermark_tokenizer", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--adversarial", type=float, default=0.0)
    parser.add_argument("--option", type=int, default=None)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--n_words_sample", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--hf_cache_dir", type=str, default="")
    args = parser.parse_args()
    print("*" * 20, "\n", args, "\n", "*" * 20, "\n")
    main(args)
