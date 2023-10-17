from ..preprocess import process_transformers
from transformers import PreTrainedTokenizer
from datasets import load_dataset, DatasetDict

MAX_LENGTH = 416 # Default max length for Wangchan-based models
# MAX_LENGTH = 510 # Maximum possible length for Wangchan-based models
OTHER_MODEL_TO_MAX_LENGTH = {
    "xlm-roberta-base": 512,
    "bert-base-multilingual-cased": 512
}


def get_raw_downstream_dataset(name: str) -> DatasetDict:
    if name in ("wisesight_sentiment", "generated_reviews_enth", "prachathai67k"):
        return load_dataset(name, cache_dir="cache")
    elif name in ("lst20_pos", "lst20_ner"):
        return load_dataset("lst20", data_dir="LST20_Corpus", cache_dir="cache")
    elif name == "wongnai_reviews":
        unsplit = load_dataset("wongnai_reviews", cache_dir="cache")
        train_and_validation = unsplit["train"].train_test_split(train_size=36000, seed=2020)
        return DatasetDict(
            train=train_and_validation["train"],
            validation=train_and_validation["test"],
            test=unsplit["test"]
        )
    elif name == "yelp_review_full":
        unsplit = load_dataset("yelp_review_full", cache_dir="cache")
        train_and_validation = unsplit["train"].train_test_split(test_size=50000, seed=2020)
        return DatasetDict(
            train=train_and_validation["train"],
            validation=train_and_validation["test"],
            test=unsplit["test"],
        )
    elif name == "thainer":
        unsplit = load_dataset("thainer", cache_dir="cache")
        labels: list = unsplit["train"].features["ner_tags"].feature.names
        old_id2label = dict(enumerate(labels))
        labels.remove("B-ไม่ยืนยัน")
        labels.remove("I-ไม่ยืนยัน")
        label2new_id = {label: id for id, label in enumerate(labels)}
        unsplit = unsplit.map(
            lambda examples: {
                "ner_tags": [
                    [
                        label2new_id.get(old_id2label[id], label2new_id["O"])
                        for id in ids
                    ] for ids in examples["ner_tags"]
                ]
            },
            batched=True
        )
        train_and_the_rest = unsplit["train"].train_test_split(train_size=5079, seed=2020)
        validation_and_test = train_and_the_rest["test"].train_test_split(train_size=635, seed=2020)
        return DatasetDict(
            train=train_and_the_rest["train"],
            validation=validation_and_test["train"],
            test=validation_and_test["test"]
        )
    elif name.startswith("thai_nner_layer_"):
        unsplit = load_dataset("thai_nner", data_dir="ThaiNNER_Corpus", cache_dir="cache")
        train_and_validation = unsplit["train"].train_test_split(test_size=len(unsplit["test"]), seed=2020)
        return DatasetDict(
            train=train_and_validation["train"],
            validation=train_and_validation["test"],
            test=unsplit["test"]
        )
    else:
        raise ValueError(f"Invalid downstream dataset name: {name}")

def get_downstream_dataset(name: str, tokenizer: PreTrainedTokenizer):
    """
    For fine-tuning of Wangchan-based models, which require special preprocessing.
    """
    dataset = get_raw_downstream_dataset(name)
    id2label: dict[int, str]
    if name == "wisesight_sentiment":
        dataset = dataset.map(
            lambda examples: tokenizer(
                [process_transformers(text) for text in examples["texts"]],
                truncation=True,
                max_length=MAX_LENGTH
            ),
            batched=True,
            remove_columns="texts"
        )
        dataset = dataset.rename_column("category", "labels")
        id2label = dict(enumerate(dataset["train"].features["labels"].names))
    elif name == "generated_reviews_enth":
        dataset = dataset.map(
            lambda examples: tokenizer(
                [process_transformers(pair["th"]) for pair in examples["translation"]],
                truncation=True,
                max_length=MAX_LENGTH
            ),
            batched=True,
            remove_columns=["translation", "correct"]
        )
        dataset = dataset.map(
            lambda examples: {"labels": [label - 1 for label in examples["review_star"]]},
            batched=True,
            remove_columns="review_star"
        )
        id2label = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    elif name == "wongnai_reviews":
        dataset = dataset.map(
            lambda examples: tokenizer(
                [process_transformers(text) for text in examples["review_body"]],
                truncation=True,
                max_length=MAX_LENGTH
            ),
            batched=True,
            remove_columns="review_body"
        )
        dataset = dataset.rename_column("star_rating", "labels")
        id2label = dict(enumerate(dataset["train"].features["labels"].names))
    elif name == "yelp_review_full":
        dataset = dataset.map(
            lambda examples: tokenizer(
                [process_transformers(text) for text in examples["text"]],
                truncation=True,
                max_length=MAX_LENGTH
            ),
            batched=True,
            remove_columns="text"
        )
        dataset = dataset.rename_column("label", "labels")
        dataset = DatasetDict(
            train=dataset["train"].select(range(36000)),
            validation=dataset["validation"].select(range(4000)),
            test=dataset["test"].select(range(6200))
        )
        id2label = dict(enumerate(dataset["train"].features["labels"].names))
    elif name == "prachathai67k":
        dataset = dataset.map(
            lambda examples: tokenizer(
                [process_transformers(text) for text in examples["title"]],
                truncation=True,
                max_length=MAX_LENGTH
            ),
            batched=True,
            remove_columns=["url", "date", "title", "body_text"]
        )
        tags = [
            "politics",
            "human_rights",
            "quality_of_life",
            "international",
            "social",
            "environment",
            "economics",
            "culture",
            "labor",
            "national_security",
            "ict",
            "education"
        ]
        dataset = dataset.map(
            lambda examples: {
                "labels": [
                    list(map(float, label))
                    for label in zip(*(examples[tag] for tag in tags))
                ]
            },
            batched=True,
            remove_columns=tags
        )
        id2label = dict(enumerate(tags))
    elif name == "thainer":
        dataset = dataset.map(
            lambda examples: {"tokens": [['<_>' if token == ' ' else token for token in tokens] for tokens in examples["tokens"]]},
            batched=True
        )
        dataset = dataset.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, ["ner_tags", "pos_tags"]),
            batched=True,
            remove_columns=["id", "tokens"]
        )
        dataset = dataset.remove_columns("pos_tags").rename_column("ner_tags", "labels")
        labels: list = dataset["train"].features["labels"].feature.names
        labels.remove("B-ไม่ยืนยัน")
        labels.remove("I-ไม่ยืนยัน")
        id2label = dict(enumerate(labels))
    elif name in ("lst20_pos", "lst20_ner"):
        dataset = dataset.map(
            lambda examples: {"tokens": [['<_>' if token == '_' else token for token in tokens] for tokens in examples["tokens"]]},
            batched=True
        )
        dataset = dataset.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, ["ner_tags", "pos_tags"]),
            batched=True,
            remove_columns=["clause_tags", "fname", "id", "tokens"]
        )
        if name == "lst20_pos":
            dataset = dataset.remove_columns("ner_tags").rename_column("pos_tags", "labels")
            id2label = dict(enumerate(dataset["train"].features["labels"].feature.names))
        else:
            dataset = dataset.remove_columns("pos_tags").rename_column("ner_tags", "labels")
            id2label = {i: label.replace('_', '-') for i, label in enumerate(dataset["train"].features["labels"].feature.names)}
    elif name.startswith("thai_nner_layer_"):
        dataset = dataset.map(
            lambda examples: {"tokens": [['<_>' if token == '_' else token for token in tokens] for tokens in examples["tokens"]]},
            batched=True
        )
        possible_layers = [f"layer_{i}" for i in range(1, 9)]
        dataset = dataset.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, possible_layers),
            batched=True,
            remove_columns="tokens"
        )
        target_layer = name.lstrip("thai_nner_")
        dataset = dataset.remove_columns(
            [layer for layer in possible_layers if layer != target_layer]
        ).rename_column(target_layer, "labels")
        id2label = dict(enumerate(dataset["train"].features["labels"].feature.names))
    else:
        raise ValueError(f"Invalid downstream dataset name: {name}")
    dataset.name = name
    return dataset, id2label

def get_downstream_dataset_no_special_preprocessing(
    name: str,
    tokenizer: PreTrainedTokenizer
):
    """
    For fine-tuning of non-Wangchan-based models, which do not require special preprocessing.
    """
    max_length = OTHER_MODEL_TO_MAX_LENGTH[tokenizer.name_or_path]
    dataset = get_raw_downstream_dataset(name)
    id2label: dict[int, str]
    if name == "wisesight_sentiment":
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["texts"],
                truncation=True,
                max_length=max_length
            ),
            batched=True,
            remove_columns="texts"
        )
        dataset = dataset.rename_column("category", "labels")
        id2label = dict(enumerate(dataset["train"].features["labels"].names))
    elif name == "generated_reviews_enth":
        dataset = dataset.map(
            lambda examples: tokenizer(
                [pair["th"] for pair in examples["translation"]],
                truncation=True,
                max_length=max_length
            ),
            batched=True,
            remove_columns=["translation", "correct"]
        )
        dataset = dataset.map(
            lambda examples: {"labels": [label - 1 for label in examples["review_star"]]},
            batched=True,
            remove_columns="review_star"
        )
        id2label = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    elif name == "wongnai_reviews":
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["review_body"],
                truncation=True,
                max_length=max_length
            ),
            batched=True,
            remove_columns="review_body"
        )
        dataset = dataset.rename_column("star_rating", "labels")
        id2label = dict(enumerate(dataset["train"].features["labels"].names))
    elif name == "yelp_review_full":
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length
            ),
            batched=True,
            remove_columns="text"
        )
        dataset = dataset.rename_column("label", "labels")
        dataset = DatasetDict(
            train=dataset["train"].select(range(36000)),
            validation=dataset["validation"].select(range(4000)),
            test=dataset["test"].select(range(6200))
        )
        id2label = dict(enumerate(dataset["train"].features["labels"].names))
    elif name == "prachathai67k":
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["title"],
                truncation=True,
                max_length=max_length
            ),
            batched=True,
            remove_columns=["url", "date", "title", "body_text"]
        )
        tags = [
            "politics",
            "human_rights",
            "quality_of_life",
            "international",
            "social",
            "environment",
            "economics",
            "culture",
            "labor",
            "national_security",
            "ict",
            "education"
        ]
        dataset = dataset.map(
            lambda examples: {
                "labels": [
                    list(map(float, label))
                    for label in zip(*(examples[tag] for tag in tags))
                ]
            },
            batched=True,
            remove_columns=tags
        )
        id2label = dict(enumerate(tags))
    elif name == "thainer":
        dataset = dataset.map(
            lambda examples: tokenize_and_align_labels_no_special_preprocessing(
                examples, tokenizer, ["ner_tags", "pos_tags"], max_length
            ),
            batched=True,
            remove_columns=["id", "tokens"]
        )
        dataset = dataset.remove_columns("pos_tags").rename_column("ner_tags", "labels")
        labels: list = dataset["train"].features["labels"].feature.names
        labels.remove("B-ไม่ยืนยัน")
        labels.remove("I-ไม่ยืนยัน")
        id2label = dict(enumerate(labels))
    elif name in ("lst20_pos", "lst20_ner"):
        dataset = dataset.map(
            lambda examples: {"tokens": [[' ' if token == '_' else token for token in tokens] for tokens in examples["tokens"]]},
            batched=True
        )
        dataset = dataset.map(
            lambda examples: tokenize_and_align_labels_no_special_preprocessing(
                examples, tokenizer, ["ner_tags", "pos_tags"], max_length
            ),
            batched=True,
            remove_columns=["clause_tags", "fname", "id", "tokens"]
        )
        if name == "lst20_pos":
            dataset = dataset.remove_columns("ner_tags").rename_column("pos_tags", "labels")
            id2label = dict(enumerate(dataset["train"].features["labels"].feature.names))
        else:
            dataset = dataset.remove_columns("pos_tags").rename_column("ner_tags", "labels")
            id2label = {i: label.replace('_', '-') for i, label in enumerate(dataset["train"].features["labels"].feature.names)}
    elif name.startswith("thai_nner_layer_"):
        dataset = dataset.map(
            lambda examples: {"tokens": [[' ' if token == '_' else token for token in tokens] for tokens in examples["tokens"]]},
            batched=True
        )
        possible_layers = [f"layer_{i}" for i in range(1, 9)]
        dataset = dataset.map(
            lambda examples: tokenize_and_align_labels_no_special_preprocessing(
                examples, tokenizer, possible_layers, max_length
            ),
            batched=True,
            remove_columns="tokens"
        )
        target_layer = name.lstrip("thai_nner_")
        dataset = dataset.remove_columns(
            [layer for layer in possible_layers if layer != target_layer]
        ).rename_column(target_layer, "labels")
        id2label = dict(enumerate(dataset["train"].features["labels"].feature.names))
    else:
        raise ValueError(f"Invalid downstream dataset name: {name}")
    dataset.name = name
    return dataset, id2label

def tokenize_and_align_labels(
    examples: dict[str],
    tokenizer: PreTrainedTokenizer,
    tags_to_align: list[str]
):
    tokenized_inputs = tokenizer(
        [
            [process_transformers(token) for token in tokens]
            for tokens in examples["tokens"]
        ],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_overflowing_tokens=True
    )
    for tag in tags_to_align:
        labels = []
        for i, sample_id in enumerate(tokenized_inputs["overflow_to_sample_mapping"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(examples[tag][sample_id][word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs[tag] = labels
    del tokenized_inputs["overflow_to_sample_mapping"]
    return tokenized_inputs

def tokenize_and_align_labels_no_special_preprocessing(
    examples: dict[str],
    tokenizer: PreTrainedTokenizer,
    tags_to_align: list[str],
    max_length: int
):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True
    )
    for tag in tags_to_align:
        labels = []
        for i, sample_id in enumerate(tokenized_inputs["overflow_to_sample_mapping"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(examples[tag][sample_id][word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs[tag] = labels
    del tokenized_inputs["overflow_to_sample_mapping"]
    return tokenized_inputs
