from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from MAGE.deployment import preprocess
from runtime_utils import resolve_runtime_device, resolve_torch_dtype, save_jsonl


GUIDANCE_CLASSIFIER_CHOICES = ["mage", "openai_roberta_base", "openai_roberta_large", "radar"]
DEPLOY_CLASSIFIER_CHOICES = [
    "mage",
    "openai_roberta_base",
    "openai_roberta_large",
    "radar",
    "kgw_wm",
    "uni_wm",
    "fastdetectgpt",
    "gltr",
]


def _load_sequence_classifier(model_dir: str, device: str):
    if device == "cuda":
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, device_map="auto")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
    model.eval()
    return model


def _load_causal_lm(model_name: str, device: str, precision: str):
    torch_dtype = resolve_torch_dtype(device, precision)
    model_kwargs = {"torch_dtype": torch_dtype}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    return model


def _resolve_watermark_tokenizer_path(
    watermark_tokenizer: str | None = None,
    hf_cache_dir: str | None = None,
) -> str:
    if watermark_tokenizer:
        return watermark_tokenizer

    if hf_cache_dir:
        return str(Path(hf_cache_dir) / "Llama-3.1-8B-Instruct")

    raise ValueError(
        "Watermark detection requires --watermark_tokenizer or --hf_cache_dir pointing to a compatible tokenizer."
    )


class MAGEDetector:
    @torch.no_grad()
    def __init__(self, device: str = "auto"):
        self.device = resolve_runtime_device(device)
        model_dir = "yaful/MAGE"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = _load_sequence_classifier(model_dir, self.device)

    @torch.no_grad()
    def get_scores(self, texts, deploy: bool = False):
        texts_ = [preprocess(text) for text in texts]
        toks = self.tokenizer(texts_, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**toks)
        scores = torch.softmax(outputs.logits, dim=-1)
        return scores[:, 0].detach().cpu().numpy()


class OpenAIRoberta:
    @torch.no_grad()
    def __init__(self, model_name: str = "openai_roberta_base", device: str = "auto"):
        self.device = resolve_runtime_device(device)
        size = model_name.split("_")[-1]
        model_dir = f"openai-community/roberta-{size}-openai-detector"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = _load_sequence_classifier(model_dir, self.device)

    @torch.no_grad()
    def get_scores(self, texts=["Hello world! Is this content AI-generated?"], deploy: bool = False):
        tokenized_input = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        logits = self.model(**tokenized_input).logits
        scores = torch.softmax(logits, dim=1)
        return scores[:, 0].detach().cpu().numpy()


class RADAR:
    @torch.no_grad()
    def __init__(self, device: str = "auto"):
        self.device = resolve_runtime_device(device)
        model_dir = "TrustSafeAI/RADAR-Vicuna-7B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, model_max_length=512)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = _load_sequence_classifier(model_dir, self.device)

    @torch.no_grad()
    def get_scores(self, texts, deploy: bool = False):
        tokenized_input = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        logits = self.model(**tokenized_input).logits
        scores = torch.softmax(logits, dim=1)
        return scores[:, 0].detach().cpu().numpy()


def build_guidance_classifier(classifier_name: str, device: str = "auto"):
    if classifier_name == "mage":
        return MAGEDetector(device=device)
    if classifier_name in {"openai_roberta_base", "openai_roberta_large"}:
        return OpenAIRoberta(model_name=classifier_name, device=device)
    if classifier_name == "radar":
        return RADAR(device=device)
    raise ValueError(f"Unsupported guidance classifier '{classifier_name}'.")


def build_watermark_detector(
    deploy_classifier: str,
    device: str = "auto",
    hf_cache_dir: str | None = None,
    watermark_tokenizer: str | None = None,
):
    resolved_device = resolve_runtime_device(device)
    tokenizer_path = _resolve_watermark_tokenizer_path(
        watermark_tokenizer=watermark_tokenizer,
        hf_cache_dir=hf_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if deploy_classifier == "kgw_wm":
        from kgw_wm.extended_watermark_processor import WatermarkDetector

        seeding_scheme = "selfhash"
    elif deploy_classifier == "uni_wm":
        from uni_wm.extended_watermark_processor import WatermarkDetector

        seeding_scheme = "unigram"
    else:
        raise ValueError(f"Unsupported watermark detector '{deploy_classifier}'.")

    return WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        seeding_scheme=seeding_scheme,
        device=resolved_device,
        tokenizer=tokenizer,
        z_threshold=4.0,
        normalizers=[],
        ignore_repeated_ngrams=True,
    )


def build_deploy_classifier(
    classifier_name: str,
    device: str = "auto",
    hf_cache_dir: str | None = None,
    watermark_tokenizer: str | None = None,
):
    if classifier_name in {"kgw_wm", "uni_wm"}:
        return build_watermark_detector(
            deploy_classifier=classifier_name,
            device=device,
            hf_cache_dir=hf_cache_dir,
            watermark_tokenizer=watermark_tokenizer,
        )

    if classifier_name == "mage":
        return MAGEDetector(device=device)
    if classifier_name in {"openai_roberta_base", "openai_roberta_large"}:
        return OpenAIRoberta(model_name=classifier_name, device=device)
    if classifier_name == "radar":
        return RADAR(device=device)
    if classifier_name in {"fastdetectgpt", "gltr"}:
        from zs_detectors.detector import get_detector

        return get_detector(classifier_name)

    raise ValueError(f"Unsupported deploy classifier '{classifier_name}'.")


class Paraphraser:
    @torch.no_grad()
    def __init__(
        self,
        name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        classifier=None,
        device: str = "auto",
        precision: str = "auto",
        top_k: int = 50,
    ):
        self.device = resolve_runtime_device(device)
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = _load_causal_lm(self.name, device=self.device, precision=precision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.generation_config.top_p = 0.99
        self.model.generation_config.top_k = top_k
        self.model.generation_config.temperature = 0.6
        self.classifier = classifier

    @torch.no_grad()
    def paraphrase(
        self,
        contents,
        batch_size: int = 10,
        max_new_tokens: int = 2048,
        top_p: float = 0.9,
        adversarial: float = 1.0,
        option: int | None = None,
        deterministic: bool = True,
    ):
        if bool(adversarial) and self.classifier is None:
            raise ValueError("Adversarial paraphrasing requires a guidance classifier.")

        system_prompt = (
            "You are a rephraser. Given any input text, you are supposed to rephrase the text without "
            "changing its meaning and content, while maintaining the text quality. Also, it is important "
            "for you to output a rephrased text that has a different style from the input text. You can "
            "not just make a few changes to the input text. The input text is given below. Print your "
            "rephrased output text between tags <TAG> and </TAG>."
        )
        self.model.generation_config.top_p = top_p

        responses = []
        for b in range(0, len(contents), batch_size):
            inputs = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for content in contents[b : b + batch_size]
            ]
            inputs = [inp + "<TAG> " for inp in inputs]
            tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)
            input_ids = tokenized_inputs["input_ids"]
            attention_mask = tokenized_inputs["attention_mask"]
            past_key_values = None
            finished = torch.zeros(len(input_ids), dtype=torch.bool, device=self.device)
            generated_tokens = [[] for _ in range(len(input_ids))]

            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                probs = torch.softmax(logits.float() / self.model.generation_config.temperature, dim=-1)

                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                mask = probs_sum - probs_sort > self.model.generation_config.top_p
                probs_sort[mask] = 0.0

                if option == 1:
                    assert 0 < adversarial <= 1
                    mask = probs_sort < adversarial
                    max_indices = probs_sort.argmax(dim=1, keepdim=True)
                    mask.scatter_(1, max_indices, False)
                    probs_sort[mask] = 0.0

                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                next_tokens, prob_scores = [], []
                for i in range(len(probs_idx)):
                    next_tokens.append(
                        probs_idx[i][probs_sort[i] > 0.0].detach().cpu().numpy().tolist()[
                            : self.model.generation_config.top_k
                        ]
                    )
                    prob_scores.append(
                        probs_sort[i][probs_sort[i] > 0.0].detach().cpu().numpy().tolist()[
                            : self.model.generation_config.top_k
                        ]
                    )

                sampled_tokens = []
                if bool(adversarial):
                    for i in range(len(next_tokens)):
                        if len(next_tokens[i]) == 1:
                            sampled_tokens.append(next_tokens[i][0])
                            continue

                        toks = torch.tensor(next_tokens[i], device=self.device).unsqueeze(1)
                        inps = (
                            torch.tensor(generated_tokens[i], dtype=input_ids.dtype)
                            .unsqueeze(0)
                            .expand(toks.shape[0], -1)
                            .to(self.device)
                        )
                        next_toks = torch.cat([inps, toks], dim=-1)
                        next_words = self.tokenizer.batch_decode(next_toks, skip_special_tokens=True)
                        adv_scores = []
                        for j in range(0, len(next_words), batch_size):
                            adv_scores.extend(self.classifier.get_scores(next_words[j : j + batch_size]))

                        if option == 2:
                            adv_scores = -np.array(prob_scores[i]) + float(adversarial) * np.array(adv_scores)

                        if deterministic:
                            idx = int(np.argmin(adv_scores))
                            sampled_tokens.append(next_tokens[i][idx])
                        else:
                            weights = -np.array(adv_scores, dtype=np.float64)
                            weights += -weights.min() + 1e-9
                            if weights.sum() <= 0:
                                raise RuntimeError("Adversarial sampling produced a zero-probability distribution.")
                            weights /= weights.sum()
                            idx = int(np.random.choice(len(next_tokens[i]), p=weights))
                            sampled_tokens.append(next_tokens[i][idx])
                else:
                    for i in range(len(next_tokens)):
                        weights = np.array(prob_scores[i], dtype=np.float64)
                        if weights.sum() <= 0:
                            raise RuntimeError("Token sampling produced a zero-probability distribution.")
                        weights /= weights.sum()
                        sampled_tokens.append(int(np.random.choice(next_tokens[i], p=weights)))

                for i in range(len(input_ids)):
                    if finished[i]:
                        continue
                    generated_tokens[i].append(sampled_tokens[i])
                    if sampled_tokens[i] == self.tokenizer.eos_token_id:
                        finished[i] = True

                if finished.all():
                    break

                input_ids = torch.tensor(sampled_tokens, dtype=input_ids.dtype).unsqueeze(1).to(self.device)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((len(input_ids), 1), dtype=torch.long, device=self.device),
                    ],
                    dim=1,
                )

            for i in range(len(input_ids)):
                response_text = self.tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
                response_text = response_text.replace("<TAG>", "").replace("</TAG>", "").strip()
                weirds = ["Note: I rephrased", "Note: I've rephrased", "Note: I have rephrased", "(Note:"]
                for weird in weirds:
                    if weird in response_text:
                        response_text = response_text.split(weird)[0].strip()
                        break
                responses.append(response_text)

        return responses
