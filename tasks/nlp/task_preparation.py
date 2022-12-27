# prepare each task encoder, decoder, custom dataset class

from datasets import load_dataset
from tomlkit import load

import os, sys
import logging

nlp_task_types = [
    "CausalLM",
    "Seq2SeqLM",
    "DocumentQuestionAnswering",
    "MaskedLM",
    "NextSentencePrediction",
    "QuestionAnswering",
    "SequenceClassification",
    "TokenClassification",
]

task_config_file = f"{os.path.dirname(os.path.abspath(__file__))}/task_config.toml"
dataset_list = []

with open(task_config_file, "r") as f:
    task_config = load(f)

for task in task_config["tasks"]:
    # check config validation
    assert "task_types" in task
    assert len(task["task_types"]) > 0
    if "sub_tasks" in task:
        assert len(task["sub_tasks"]) == len(task["task_types"])
    if (
        "SequenceClassification" in task["task_types"]
        or "TokenClassification" in task["task_types"]
    ):
        assert "label_keys" in task
        assert len(task["label_keys"]) == len(task["task_types"])

    # dataset download part
    if "sub_tasks" in task:
        for sub_task in task["sub_tasks"]:
            dataset = load_dataset(task["name"], sub_task)
    elif "languages" in task:
        for language in task["languages"]:
            dataset = load_dataset(task["name"], language)
    else:
        dataset = load_dataset(task["name"])

    dataset_list.append(dataset)

    # generate each task encoder / decoder
