import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, create_optimizer
from rouge_score import rouge_scorer

class FinSumm:
    def __init__(self):
        self.dirname = os.path.dirname(__file__)
        self.LR = 4e-5
        self.BATCH_SIZE = 8
        self.EPOCHS = 10
        self.WEIGHT_DECAY = 0.01
        self.max_input_length = 512
        self.max_target_length = 128
        self.trained = False
        self.model_name = 'runaksh/financial_summary_T5_base'
        self.data_file = os.path.join(self.dirname, 'data/text_with_summary.xlsx')
        self.data = None
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.__score__ = 0.5
        self.load()

    def load(self, data_path=None):
        try:
            self.model = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.trained = True
        except:
            if os.path.exists(os.path.join(self.dirname, 'financial_summary_T5_base')):
                self.model_name = os.path.join(self.dirname, 'financial_summary_T5_base')

            self.trained = self.model_name != 't5-base'

            self.model = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if data_path:
            self.data_file = data_path
        df = pd.read_excel(self.data_file)
        self.data = df[['Text', 'Summary']]
        self.loaded = True

    # creating a function to preprocess (convert tokens into ids) the input and target sequences
    def preprocess(self, data):
        if not self.loaded:
            self.load()
        inputs = self.tokenizer(
            data['Text'],
            max_length=self.max_input_length,
            truncation=True
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                data['Summary'],
                max_length=self.max_target_length,
                truncation=True
            )
        inputs['labels'] = labels['input_ids']
        return inputs


    def train(self, data_path=None, train_size=0.8, force=False, refresh=False):
        if refresh:
            self.model_name = 't5_base'
            self.load(data_path)
        if not self.loaded:
            self.load(data_path)
        if (not self.trained) or force:
            # because the T5 model has been pretrained in the similar fashion for summarization tasks
            self.data['Text'] = self.data['Text'].apply(lambda x: "summarize: " + x)
            split = int(len(self.data) * train_size)
            training_data = self.data[:split]
            valid_data = self.data[split:]
            dataset=DatasetDict()
            dataset['training']=Dataset.from_pandas(training_data)
            dataset['validation']=Dataset.from_pandas(valid_data)
            tokenized_dataset = dataset.map(self.preprocess, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(['Summary', 'Text'])
            datacollator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, return_tensors="tf")

            # for training
            tf_train_dataset = tokenized_dataset['training'].to_tf_dataset(
                columns=['input_ids', 'attention_mask', 'labels'],
                collate_fn=datacollator,
                shuffle=True,
                batch_size=self.BATCH_SIZE)

            # for validation
            tf_valid_dataset = tokenized_dataset['validation'].to_tf_dataset(
                columns=['input_ids', 'attention_mask', 'labels'],
                collate_fn=datacollator,
                shuffle=False,
                batch_size=self.BATCH_SIZE)

            num_train_steps = len(tf_train_dataset) * self.EPOCHS

            optimizer, schedule = create_optimizer(
                init_lr=self.LR,
                num_warmup_steps=0,
                num_train_steps=num_train_steps,
                weight_decay_rate=self.WEIGHT_DECAY,
            )

            self.model.compile(optimizer=optimizer)  # for loss the model will use the model's internal loss by default

            # Training in mixed-precision float16 for faster training and efficient memory usage
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

            self.model.fit(tf_train_dataset, validation_data=tf_valid_dataset, epochs=self.EPOCHS)
            self.model.save_pretrained("financial_summary_T5_base")
            self.tokenizer.save_pretrained("financial_summary_T5_base")
            self.trained = True
            self.loaded = False

    def generate_summary(self, text, min_length=55, max_length=80):
        text = "summarize: " + text
        input_txt = self.tokenizer(text, max_length=self.max_input_length, truncation=True, return_tensors="tf").input_ids
        op = self.model.generate(input_txt, min_length=min_length, max_length=max_length)
        decoded_op = self.tokenizer.batch_decode(op, skip_special_tokens=True)
        return decoded_op[0]

    def metric(self):
        sut = self.data.sample()
        pred = self.generate_summary(sut["Text"].values[0])
        self.__score__ = self.scorer.score(pred, sut["Summary"].values[0])
        return self.__score__['rouge1'].precision
