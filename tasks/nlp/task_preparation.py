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
    "Regression",
]


def get_config_dataset():
    task_config_file = f"{os.path.dirname(os.path.abspath(__file__))}/task_config.toml"
    dataset_list = []

    with open(task_config_file, "r") as f:
        task_config = load(f)

    for task in task_config["tasks"]:
        # check config validation
        logging.debug(f"preparing {task['name']} ...")

        if "source" in task and task["source"] == "korpora":
            continue

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
            assert len(task["feature_keys"]) == len(task["task_types"])

        # dataset download part
        if "sub_tasks" in task:
            for i, sub_task in enumerate(task["sub_tasks"]):
                dataset = load_dataset(task["name"], sub_task)
                dataset["name"] = f"{task['name']}-{sub_task}"
                dataset["task_type"] = task["task_types"][i]
                dataset["feature_keys"] = task["feature_keys"][i]
                dataset["label_key"] = task["label_keys"][i]
                dataset_list.append(dataset)
        elif "languages" in task:
            for language in task["languages"]:
                dataset = load_dataset(task["name"], language)
        else:
            dataset = load_dataset(task["name"])
            dataset["name"] = f"{task['name']}"
            dataset["task_type"] = task["task_types"][0]
            dataset["feature_keys"] = task["feature_keys"][0]
            dataset["label_key"] = task["label_keys"][0]
            dataset_list.append(dataset)

    return dataset_list
