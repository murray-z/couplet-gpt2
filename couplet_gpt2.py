import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, GPT2LMHeadModel

gpt_model_name = "uer/gpt2-distil-chinese-cluecorpussmall"
train_data_path = "./data/couplet/train"
test_data_path = "./data/couplet/test"


tokenizers = BertTokenizer.from_pretrained(gpt_model_name)

batch = 32
epochs = 3
lr = 2e-5
max_len = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model_path = "./pytorch_model.pt"


class CoupletDataset(Dataset):
    def __init__(self, data_dir):
        self.texts = []
        in_path = os.path.join(data_dir, "in.txt")
        out_path = os.path.join(data_dir, "out.txt")
        with open(in_path, "r", encoding="utf-8") as f:
            ins = f.readlines()
        with open(out_path, "r", encoding="utf-8") as f:
            outs = f.readlines()

        for i, o in zip(ins, outs):
            i = i.strip().replace(" ", "")
            if len(i) in [5, 7]:
                self.texts.append(i + "-" + o.strip().replace(" ", ""))

    def __getitem__(self, idx):
        text = self.texts[idx]
        res = tokenizers.encode_plus(text, padding="max_length", return_tensors="pt",
                                     truncation=True, max_length=max_len, add_special_tokens=True)
        return res["input_ids"], res["attention_mask"], res["token_type_ids"]

    def __len__(self):
        return len(self.texts)


def train():
    model = GPT2LMHeadModel.from_pretrained(gpt_model_name)
    model.to(device)
    optimizer = AdamW(params=model.parameters(), lr=lr)
    train_dataset = CoupletDataset(train_data_path)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch)

    for epoch in range(epochs):
        model.train()
        for step, data in enumerate(train_dataloader):
            input_ids, attention_mask, token_type_ids = [d.to(device) for d in data]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print("Epoch: {} Step: {} Loss: {:.4f}".format(epoch, step, loss.item()))

    torch.save(model.state_dict(), save_model_path)


if __name__ == '__main__':
    train()





