import argparse
import evaluate
from seqeval import metrics
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer

# 加载数据集
#datasets = load_dataset("peoples_daily_ner")
# data_file = "./data_file/train.json"data_file

LABELS = list("BIESO")
ID2LABEL = {str(idx):label for (idx, label) in enumerate(LABELS)}

data_file = "./data_file/train.json"
model_name = "nghuyong/ernie-3.0-base-zh"  # 所使用模型
dataset = load_dataset("json", data_files=data_file, cache_dir='cache')
dataset = dataset["train"]
#eval_dataset = dataset.train_test_split(.1)["test"]
#label_list = datasets["train"].features["ner_tags"].feature.names
#label_list = datasets["train"].features["ner_tags"]
# 数据集处理
#tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh",cache_dir='cache')

def tags2feature(tags):
    return [LABELS.index(x) for x in list(tags)]

def process_function(examples):
    sources = [list(x) for x in examples["source"]]
    tags = [tags2feature(x) for x in examples["tags"]]

    tokenized_examples = tokenizer(sources, truncation=True, is_split_into_words=True, max_length=256)
    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_examples.word_ids(batch_index=i)  
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)
    tokenized_examples["labels"] = labels
    return tokenized_examples
tokenized_datasets = dataset.map(process_function, batched=True)
dataset = tokenized_datasets
eval_dataset = dataset.train_test_split(.1)["test"]

# 构建评估函数
# seqeval_metric = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = [
        [LABELS[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABELS[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # results = seqeval_metric.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")
    return {
        "precision": metrics.precision_score(true_labels, true_predictions),
        "recall": metrics.recall_score(true_labels, true_predictions),
        "f1": metrics.f1_score(true_labels, true_predictions),
        "accuracy": metrics.accuracy_score(true_labels, true_predictions),
    }

# 配置训练器
model = AutoModelForTokenClassification.from_pretrained("nghuyong/ernie-3.0-base-zh", num_labels=len(LABELS), id2label=ID2LABEL,cache_dir='cache')
args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    num_train_epochs=2,
    weight_decay=0.01,
    output_dir="model_for_tokenclassification",
    logging_steps=1000,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
trainer = Trainer(
    model,
    args,
    # train_dataset=tokenized_datasets["dataset"],
    train_dataset=dataset,
    # eval_dataset=tokenized_datasets["dataset"],
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 训练与评估
trainer.train()
#trainer.evaluate(tokenized_datasets["test"])
trainer.evaluate()


