import os
from typing import Tuple, List

import click
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertModel,
    AdamW,
    BertPreTrainedModel,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


BATCH_SIZE = 32
CHECKPOINTS_DIR = "./checkpoints"
DATA_PATH = "./jigsaw-toxic-comment-classification-challenge/"
NUM_EPOCHS = 2

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")


# Here we create a Dataset and iterators to it for training and validation. For
# very big datasets it can be done in a lazy way, however I choose to load all
# the data at once.
class ToxicDataset(Dataset):
    def __init__(self, tokenizer, dataframe, device):
        self.device = device
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id
        self.X = []
        self.Y = []
        for i, (row) in tqdm(dataframe.iterrows()):
            if len(tokenizer.tokenize(row["comment_text"])) > 120:
                continue
            text = tokenizer.encode(row["comment_text"], add_special_tokens=True)
            text = torch.LongTensor(text)
            tags = torch.FloatTensor(
                row[
                    [
                        "toxic",
                        "severe_toxic",
                        "obscene",
                        "threat",
                        "insult",
                        "identity_hate",
                    ]
                ]
            )
            self.X.append(text)
            self.Y.append(tags)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return self.X[index], self.Y[index]


# Simple Bert model for classification of whole sequence.
class BertClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 6)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        cls_output = outputs[1]  # batch, hidden
        cls_output = self.classifier(cls_output)  # batch, 6
        cls_output = torch.sigmoid(cls_output)
        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            loss = criterion(cls_output, labels)
        return loss, cls_output


def collate_fn(
    batch: List[Tuple[torch.LongTensor, torch.LongTensor]]
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    x, y = list(zip(*batch))
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x.to(DEVICE), y.to(DEVICE)


def train_epoch(model, iterator, optimizer, scheduler):
    model.train()
    total_loss = 0
    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        mask = (x != 0).float()
        loss, outputs = model(x, attention_mask=mask, labels=y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f"Train loss {total_loss / len(iterator)}")


def evaluate_epoch(model, iterator):
    model.eval()
    pred = []
    true = []
    with torch.no_grad():
        total_loss = 0
        for x, y in tqdm(iterator):
            mask = (x != 0).float()
            loss, outputs = model(x, attention_mask=mask, labels=y)
            total_loss += loss
            true += y.cpu().numpy().tolist()
            pred += outputs.cpu().numpy().tolist()
    true = np.array(true)
    pred = np.array(pred)
    for i, name in enumerate(
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ):
        print(f"{name} roc_auc {roc_auc_score(true[:, i], pred[:, i])}")
    print(f"Evaluate loss {total_loss / len(iterator)}")


def create_test_submission(model, tokenizer):
    model.eval()

    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
    columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    for i in tqdm(range(len(test_df) // BATCH_SIZE + 1)):
        batch_df = test_df.iloc[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        assert (
            batch_df["id"] == submission["id"][i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        ).all(), f"Id mismatch"

        texts = []
        for text in batch_df["comment_text"].tolist():
            text = tokenizer.encode(text, add_special_tokens=True)
            if len(text) > 120:
                text = text[:119] + [tokenizer.sep_token_id]

            texts.append(torch.LongTensor(text))

        x = pad_sequence(
            texts, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(DEVICE)
        mask = (x != tokenizer.pad_token_id).float().to(DEVICE)
        with torch.no_grad():
            _, outputs = model(x, attention_mask=mask)
        outputs = outputs.cpu().numpy()
        submission.iloc[i * BATCH_SIZE : (i + 1) * BATCH_SIZE][columns] = outputs

    submission.to_csv("submission.csv", index=False)


def train():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    assert (
        tokenizer.pad_token_id == 0
    ), "Padding value used in masks is set to zero, please change it everywhere"

    # Here we are reading the data into the DataFrame
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

    # training on a part of data for speed
    # train_df = train_df.sample(frac=0.33)
    train_df, val_df = train_test_split(train_df, test_size=0.05)

    train_dataset = ToxicDataset(tokenizer, train_df, DEVICE)
    dev_dataset = ToxicDataset(tokenizer, val_df, DEVICE)

    train_sampler = RandomSampler(train_dataset)
    dev_sampler = RandomSampler(dev_dataset)
    train_iterator = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    dev_iterator = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, sampler=dev_sampler, collate_fn=collate_fn
    )

    model = BertClassifier.from_pretrained("bert-base-cased").to(DEVICE)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # triangular learning rate, linearly grows untill half of first epoch, then linearly decays
    warmup_steps = int(0.5 * len(train_iterator))
    total_steps = len(train_iterator) * NUM_EPOCHS - warmup_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, last_epoch=-1
    )

    for i in range(NUM_EPOCHS):
        print("=" * 50, f"EPOCH {i}", "=" * 50)
        train_epoch(model, train_iterator, optimizer, scheduler)
        evaluate_epoch(model, dev_iterator)

        save_state = {
            "epoch": i,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tokenizer": tokenizer,
        }
        torch.save(save_state, os.path.join(CHECKPOINTS_DIR, f"epoch{NUM_EPOCHS}.ckpt"))

    # NOTE(brendan): Test evaluation code (with Kaggle submission preparation).
    create_test_submission(model, tokenizer)


def test(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model = BertClassifier.from_pretrained("bert-base-cased").to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    create_test_submission(model, checkpoint["tokenizer"])


@click.command()
@click.option("--do-train/--no-do-train", default=True)
@click.option("--checkpoint-path", type=str, default=None)
def main(do_train, checkpoint_path):
    if do_train:
        train()
    else:
        test(checkpoint_path)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
