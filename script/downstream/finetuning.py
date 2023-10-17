from typing import Literal
from os import path
from datasets import DatasetDict
from seqeval.metrics import classification_report as seqeval_metric
from sklearn.metrics import classification_report as sklearn_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    EvalPrediction
)
import torch

Task = Literal[
    "single_label_classification",
    "multi_label_classification",
    "token_classification",
    "named_entity_recognition"
]

def f1_metrics(
    task: Task,
    id2label: dict[int, str]
):
    # Function that gets predictions and labels from EvalPrediction
    if task == "single_label_classification":
        preprocess = lambda eval_pred: (
            eval_pred.predictions.argmax(axis=1),
            eval_pred.label_ids
        )
    elif task == "multi_label_classification":
        preprocess = lambda eval_pred: (
            torch.nn.Sigmoid()(torch.tensor(eval_pred.predictions)) > 0.5,
            eval_pred.label_ids
        )
    elif task == "named_entity_recognition":
        def preprocess(eval_pred: EvalPrediction):
            predictions = eval_pred.predictions.argmax(axis=2)
            labels = eval_pred.label_ids
            predictions = [
                [id2label[p] for p, l in zip(p_row, l_row) if l != -100]
                for p_row, l_row in zip(predictions, labels)
            ]
            labels = [
                [id2label[l] for l in l_row if l != -100]
                for l_row in labels
            ]
            return predictions, labels
    elif task == "token_classification":
        def preprocess(eval_pred: EvalPrediction):
            predictions = eval_pred.predictions.argmax(axis=2).flatten()
            labels = eval_pred.label_ids.flatten()
            predictions = [id2label[p] for p, l in zip(predictions, labels) if l != -100]
            labels = [id2label[l] for l in labels if l != -100]
            return predictions, labels
    else:
        raise ValueError(f"Invalid task: {task}")
    # Function that computes metrics
    if task == "named_entity_recognition":
        metric = seqeval_metric
    else:
        metric = sklearn_metric
    # Function that converts result to Huggingface's format
    if task == "single_label_classification":
        postprocess = lambda result: {
            "micro_average_f1": result["accuracy"],
            "macro_average_f1": result["macro avg"]["f1-score"],
            "class_f1": {
                id2label[int(i)]: result[i]["f1-score"]
                for i in result
                    if i.isdigit()
            }
        }
    elif task == "token_classification":
        tag_set = set(id2label.values())
        postprocess = lambda result: {
            "micro_average_f1": result["accuracy"],
            "macro_average_f1": result["macro avg"]["f1-score"],
            "class_f1": {
                tag: result[tag]["f1-score"]
                for tag in result
                    if tag in tag_set
            }
        }
    elif task == "named_entity_recognition":
        tag_set = {tag[2:] for tag in id2label.values() if tag != "O"}
        postprocess = lambda result: {
            "micro_average_f1": result["micro avg"]["f1-score"],
            "macro_average_f1": result["macro avg"]["f1-score"],
            "class_f1": {
                tag: result[tag]["f1-score"]
                for tag in result
                    if tag in tag_set
            }
        }
    else:
        postprocess = lambda result: {
            "micro_average_f1": result["micro avg"]["f1-score"],
            "macro_average_f1": result["macro avg"]["f1-score"],
            "class_f1": {
                id2label[int(i)]: result[i]["f1-score"]
                for i in result
                    if i.isdigit()
            }
        }
    def compute_metrics(eval_pred: EvalPrediction):
        predictions, labels = preprocess(eval_pred)
        result = metric(y_pred=predictions, y_true=labels, output_dict=True)
        return postprocess(result)
    return compute_metrics

DATASET_NAME_TO_TASK: dict[str, Task] = {
    "wisesight_sentiment": "single_label_classification",
    "generated_reviews_enth": "single_label_classification",
    "wongnai_reviews": "single_label_classification",
    "yelp_review_full": "single_label_classification",
    "prachathai67k": "multi_label_classification",
    "thainer": "named_entity_recognition",
    "lst20_pos": "token_classification",
    "lst20_ner": "named_entity_recognition",
    "thai_nner_layer_1": "named_entity_recognition",
    "thai_nner_layer_2": "named_entity_recognition",
    "thai_nner_layer_3": "named_entity_recognition",
    "thai_nner_layer_4": "named_entity_recognition",
    "thai_nner_layer_5": "named_entity_recognition",
    "thai_nner_layer_6": "named_entity_recognition",
    "thai_nner_layer_7": "named_entity_recognition",
    "thai_nner_layer_8": "named_entity_recognition",
    "wongnai_yelp": "single_label_classification"
}

def finetune_on_dataset(
    *,
    dataset: DatasetDict,
    id2label: dict[int, str],
    model_dir: str,
    tokenizer: PreTrainedTokenizer,
    do_train: bool = True,
    do_test: bool = True,
    override_default: dict[str] | None = None
):
    # Get model
    task = DATASET_NAME_TO_TASK[dataset.name]
    if task in ("named_entity_recognition", "token_classification"):
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            num_labels=len(id2label)
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            problem_type=task,
            num_labels=len(id2label)
        )
    # Data collator
    if task in ("named_entity_recognition", "token_classification"):
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Training arguments
    if task == "single_label_classification":
        metric_for_best_model = "eval_micro_average_f1"
    elif task == "multi_label_classification":
        metric_for_best_model = "eval_macro_average_f1"
    else:
        metric_for_best_model = "eval_loss"
    if dataset.name == "thainer" or dataset.name.startswith("thai_nner_layer_"):
        steps = 20
    else:
        steps = 100
    if dataset.name.startswith("thai_nner_layer_"):
        num_train_epochs = 20
    elif task in ("named_entity_recognition", "token_classification"):
        num_train_epochs = 6
    else:
        num_train_epochs = 3
    args = dict(
        output_dir=path.join("finetuned_models", dataset.name),
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=steps,
        save_strategy="steps",
        save_steps=steps,
        save_total_limit=5,
        per_device_train_batch_size=32 if task in ("named_entity_recognition", "token_classification") else 16,
        per_device_eval_batch_size=32 if task in ("named_entity_recognition", "token_classification") else 16,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        num_train_epochs=num_train_epochs,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model
    )
    if override_default is not None:
        args.update(override_default)
    training_args = TrainingArguments(**args)
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=f1_metrics(task, id2label)
    )

    if do_train:
        try:
            trainer.train()
        except KeyboardInterrupt:
            pass
    if do_test:
        print("**Evaluate on test set**")
        print('\n'.join(f"{k}: {v}" for k, v in trainer.predict(test_dataset=dataset["test"]).metrics.items()))

    return trainer
