import os.path

import pandas as pd
import numpy as np

from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
import evaluate


class FinSentiment:
    def __init__(self):
        self.dirname = os.path.dirname(__file__)
        self.LR = 2e-5
        self.BATCH_SIZE = 32
        self.EPOCHS = 10
        self.WEIGHT_DECAY = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained = False
        self.model_name = 'Runaksh/financial_sentiment_distilBERT'
        self.data = None
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.loaded = False
        self.n_classes = 3
        self.id2label = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        self.load()

    def load(self):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.trained = True
        except:
            if os.path.exists(os.path.join(self.dirname, 'financial_sentiment_distilBERT')):
                self.model_name = os.path.join(self.dirname, 'financial_sentiment_distilBERT')
            self.data = os.path.join(self.dirname, 'data/FinancialNews_India_Sentiment.csv')

            self.trained = self.model_name != 'distilbert-base-uncased'

            if self.trained:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=self.n_classes,
                    id2label=self.id2label,
                    label2id={label: idx for idx, label in self.id2label.items()}
                )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.trained:
            self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        self.loaded = True

    def train(self, data_path=None, train_size=0.8, force=False):
        if not self.loaded:
            self.load()
        if (not self.trained) or force:
            if data_path:
                self.data = data_path
            df = pd.read_csv(self.data)
            label2id = {label: idx for idx, label in self.id2label.items()}
            # label encode our labels
            df['label'] = df['label'].map(label2id)
            dataset = Dataset.from_pandas(df).train_test_split(train_size=train_size)

            # Tokenize and encode the dataset
            def tokenize(batch):
                tokenized_batch = self.tokenizer(batch['text'], padding=True, truncation=True)
                return tokenized_batch

            dataset_enc = dataset.map(tokenize, batched=True)

            accuracy = evaluate.load('accuracy')

            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                return accuracy.compute(predictions=predictions, references=labels)

            training_args = TrainingArguments(
                output_dir=os.path.join(self.dirname, '../Checkpoints'),
                learning_rate=self.LR,
                per_device_train_batch_size=self.BATCH_SIZE,
                per_device_eval_batch_size=self.BATCH_SIZE,
                num_train_epochs=self.EPOCHS,
                weight_decay=self.WEIGHT_DECAY,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                push_to_hub=False,
                report_to="none"
            )

            dc = DataCollatorWithPadding(self.tokenizer, return_tensors='pt')
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset_enc["train"],
                eval_dataset=dataset_enc["test"],
                tokenizer=self.tokenizer,
                data_collator=dc,
                compute_metrics=compute_metrics
            )

            trainer.train()
            trainer.save_model(os.path.join(self.dirname, 'financial_sentiment_distilBERT'))
            print(trainer.evaluate())
            self.trained = True
            self.loaded = False

    def predict(self, data):
        if not self.loaded:
            self.load()
        if self.trained:
            return self.classifier(data)[0]["label"]
        return "neutral"

    def metric(self, n):
        df = pd.read_csv(self.data)
        dfn = df.sample(n)
        count = 0
        for _, row in dfn.iterrows():
            count += int(row["label"] == self.predict(row["text"]))
        return count/n
