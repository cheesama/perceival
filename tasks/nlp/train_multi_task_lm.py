from task_preparation import get_config_dataset
from language_task_dataset import LanguageMultiTaskDataset
from language_task_encoder_decoder import LanguageMultiTaskEncoderDecoder

import logging

# validate config & download data
data_list = get_config_dataset()

train_datasets = []
valid_datasets = []

for entire_data in data_list:
    logging.warn(entire_data["name"])
    data = entire_data["train"]
    dataset = LanguageMultiTaskDataset(
        task_type=entire_data["task_type"],
        task_name=entire_data["name"],
        data=data,
        feature_keys=entire_data["feature_keys"],
        label_key=entire_data["label_key"],
    )
    train_datasets.append(dataset)
    logging.info(f"train dataset: {entire_data['name']} appended")

    if "validation" in entire_data:
        data = entire_data["validation"]
    elif "valid" in entire_data:
        data = entire_data["valid"]
    else:
        data = entire_data["test"]
    dataset = LanguageMultiTaskDataset(
        task_type=entire_data["task_type"],
        task_name=entire_data["name"],
        data=data,
        feature_keys=entire_data["feature_keys"],
        label_key=entire_data["label_key"],
    )
    valid_datasets.append(dataset)
    logging.info(f"valid dataset: {entire_data['name']} appended")

pl_model = LanguageMultiTaskEncoderDecoder(train_datasets, valid_datasets)

logging.info(pl_model)
