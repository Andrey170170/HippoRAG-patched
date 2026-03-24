import os
from typing import List
import json

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.StandardRAG import StandardRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.embedding_store import EmbeddingStore

import argparse

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging


def get_gold_docs(samples: List, dataset_name: str | None = None) -> List:
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:  # hotpotqa, 2wikimultihopqa
            if dataset_name is None:
                raise ValueError(
                    "dataset_name is required for supporting_facts datasets"
                )
            gold_title = set([item[0] for item in sample["supporting_facts"]])
            gold_title_and_content_list = [
                item for item in sample["context"] if item[0] in gold_title
            ]
            if dataset_name.startswith("hotpotqa"):
                gold_doc = [
                    item[0] + "\n" + "".join(item[1])
                    for item in gold_title_and_content_list
                ]
            else:
                gold_doc = [
                    item[0] + "\n" + " ".join(item[1])
                    for item in gold_title_and_content_list
                ]
        elif "contexts" in sample:
            gold_doc = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"]
                if item["is_supporting"]
            ]
        else:
            assert "paragraphs" in sample, (
                "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            )
            gold_paragraphs = []
            for item in sample["paragraphs"]:
                if "is_supporting" in item and item["is_supporting"] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [
                item["title"]
                + "\n"
                + (item["text"] if "text" in item else item["paragraph_text"])
                for item in gold_paragraphs
            ]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if "answer" in sample or "gold_ans" in sample:
            gold_ans = sample["answer"] if "answer" in sample else sample["gold_ans"]
        elif "reference" in sample:
            gold_ans = sample["reference"]
        elif "obj" in sample:
            gold_ans = set(
                [sample["obj"]]
                + [sample["possible_answers"]]
                + [sample["o_wiki_title"]]
                + [sample["o_aliases"]]
            )
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if "answer_aliases" in sample:
            gold_ans.update(sample["answer_aliases"])

        gold_answers.append(gold_ans)

    return gold_answers


def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument(
        "--llm_provider", type=str, default=None, help="LLM provider override"
    )
    parser.add_argument("--llm_base_url", type=str, default=None, help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default=None, help="LLM name")
    parser.add_argument(
        "--embedding_provider",
        type=str,
        default=None,
        help="Embedding provider override",
    )
    parser.add_argument(
        "--embedding_base_url", type=str, default=None, help="Embedding base URL"
    )
    parser.add_argument(
        "--embedding_name", type=str, default=None, help="Embedding model name"
    )
    parser.add_argument(
        "--force_index_from_scratch",
        type=string_to_bool,
        default=None,
        help="If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.",
    )
    parser.add_argument(
        "--force_openie_from_scratch",
        type=string_to_bool,
        default=None,
        help="If set to False, will try to first reuse openie results for the corpus if they exist.",
    )
    parser.add_argument(
        "--openie_mode",
        choices=["online", "offline"],
        default=None,
        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes",
    )
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory")
    args = parser.parse_args()

    config = BaseConfig(
        dataset="musique",
        llm_provider="openrouter",
        embedding_provider="openrouter",
        embedding_model_name="openai/text-embedding-3-small",
        max_qa_steps=3,
        embedding_batch_size=1,
        max_new_tokens=None,
    )
    if args.config is not None:
        config = config.merged_with(BaseConfig.load_yaml_dict(args.config))

    cli_overrides = {
        "dataset": args.dataset,
        "llm_provider": args.llm_provider,
        "llm_base_url": args.llm_base_url,
        "llm_name": args.llm_name,
        "embedding_provider": args.embedding_provider,
        "embedding_base_url": args.embedding_base_url,
        "embedding_model_name": args.embedding_name,
        "force_index_from_scratch": args.force_index_from_scratch,
        "force_openie_from_scratch": args.force_openie_from_scratch,
        "openie_mode": args.openie_mode,
        "save_dir": args.save_dir,
    }
    config = config.merged_with(cli_overrides)

    dataset_name = config.dataset
    if dataset_name is None:
        raise ValueError("dataset must be configured via defaults, YAML, or CLI")
    legacy_dataset_save_dir = (
        f"{config.save_dir}_{dataset_name}" if config.save_dir is not None else None
    )
    if (
        config.save_dir is not None
        and legacy_dataset_save_dir is not None
        and not os.path.exists(config.save_dir)
        and os.path.exists(legacy_dataset_save_dir)
    ):
        config = config.merged_with({"save_dir": legacy_dataset_save_dir})

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    # Prepare datasets and evaluation
    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json", "r"))
    all_queries = [s["question"] for s in samples]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers), (
            "Length of queries, gold_docs, and gold_answers should be the same."
        )
    except:
        gold_docs = None
    config = config.merged_with({"corpus_len": len(corpus)})

    logging.basicConfig(level=logging.INFO)

    hipporag = StandardRAG(global_config=config)
    # hipporag = HippoRAG(global_config=config)

    hipporag.index(docs)

    # Retrieval and QA
    hipporag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)


if __name__ == "__main__":
    main()
