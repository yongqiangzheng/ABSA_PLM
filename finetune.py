import logging
import argparse
import math
import os
import sys
import random
import numpy as np

from sklearn import metrics
from time import strftime, localtime

from transformers import BertTokenizer, RobertaTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class ABSADataset(Dataset):
    def __init__(self, file, tokenizer, max_len):
        self.file = file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.load_data()

    def load_data(self):
        fin = open(self.file, "r", encoding="utf-8",
                   newline="\n", errors="ignore")
        lines = fin.readlines()
        fin.close()
        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip()
                                        for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            text = text_left + " " + aspect + " " + text_right
            tensor_dict = self.tokenizer(text, aspect, padding="max_length", max_length=self.max_len,
                                         return_tensors="pt")

            if type(self.tokenizer) == BertTokenizer:
                input_ids = tensor_dict["input_ids"].squeeze(0)
                token_type_ids = tensor_dict["token_type_ids"].squeeze(0)
                attention_mask = tensor_dict["attention_mask"].squeeze(0)
                polarity = int(polarity) + 1
                data = {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "polarity": polarity
                }
            elif type(self.tokenizer) == RobertaTokenizer:
                input_ids = tensor_dict["input_ids"].squeeze(0)
                attention_mask = tensor_dict["attention_mask"].squeeze(0)
                polarity = int(polarity) + 1
                data = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "polarity": polarity
                }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Hyper Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="bert", type=str)
parser.add_argument("--dataset", default="lap14", type=str,
                    help="twitter, restaurant, laptop")
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument("--repeat", default=3, type=int)
parser.add_argument("--lr", default=2e-5, type=float,
                    help="try 5e-5, 2e-5 for BERT, 1e-3 for others")
parser.add_argument("--l2reg", default=1e-4, type=float)
parser.add_argument("--num_epoch", default=100, type=int,
                    help="try larger number for non-BERT models")
parser.add_argument("--batch_size", default=16, type=int,
                    help="try 16, 32, 64 for BERT models")
parser.add_argument("--log_step", default=10, type=int)
parser.add_argument("--bert_dim", default=768, type=int)
parser.add_argument("--max_seq_len", default=128, type=int)
parser.add_argument("--polarities_dim", default=3, type=int)
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--device", default="cuda", type=str, help="e.g. cuda:0")
parser.add_argument("--seed", default=42, type=int,
                    help="set seed for reproducibility")
parser.add_argument("--save_model_dir",
                    default="/media/b115/Backup/NLP", type=str)
opt = parser.parse_args()
if not os.path.exists("log/{}".format(opt.model_name)):
    os.makedirs("log/{}".format(opt.model_name))
log_file = "log/{}/{}-{}.log".format(opt.model_name,
                                     opt.dataset, strftime("%y%m%d-%H%M", localtime()))
logger.addHandler(logging.FileHandler(log_file))

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(opt.seed)

dataset_files = {
    "lap14": {
        "train": "./dataset/txt/lap14_train.txt",
        "test": "./dataset/txt/lap14_test.txt"
    },
    "rest14": {
        "train": "./dataset/txt/rest14_train.txt",
        "test": "./dataset/txt/rest14_test.txt"
    },
    "mams": {
        "train": "./dataset/txt/mams_train.txt",
        "dev": "./dataset/txt/mams_dev.txt",
        "test": "./dataset/txt/mams_test.txt"
    }
}


def print_args(model):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape))
        if p.requires_grad:
            n_trainable_params += n_params
        else:
            n_nontrainable_params += n_params
    logger.info(
        "> n_trainable_params: {0}, n_nontrainable_params: {1}".format(n_trainable_params, n_nontrainable_params))
    logger.info("> training arguments:")
    for arg in vars(opt):
        logger.info(">>> {0}: {1}".format(arg, getattr(opt, arg)))


def train(model, inputs_cols, criterion, optimizer, train_data_loader, val_data_loader):
    max_val_acc = 0
    max_val_f1 = 0
    max_val_epoch = 0
    global_step = 0
    path = None
    for i_epoch in range(opt.num_epoch):
        logger.info(">" * 100)
        logger.info("epoch: {}".format(i_epoch))
        n_correct, n_total, loss_total = 0, 0, 0
        # switch model to training mode
        model.train()
        for i_batch, batch in enumerate(train_data_loader):
            global_step += 1
            # clear gradient accumulators
            optimizer.zero_grad()

            inputs = {}
            for col in inputs_cols:
                inputs[col] = batch[col].to(opt.device)
            outputs = model(**inputs)
            logits = outputs.logits
            targets = batch["polarity"].to(opt.device)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            n_correct += (torch.argmax(logits, -1) == targets).sum().item()
            n_total += len(logits)
            loss_total += loss.item() * len(logits)
            if global_step % opt.log_step == 0:
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                logger.info("loss: {:.4f}, acc: {:.4f}".format(
                    train_loss, train_acc))

        val_acc, val_f1 = evaluate_acc_f1(model, inputs_cols, val_data_loader)
        logger.info(
            "> val_acc: {:.4f}, val_f1: {:.4f}".format(val_acc, val_f1))

        if val_acc > max_val_acc:  # acc improve
            max_val_acc = val_acc
            max_val_f1 = val_f1
            max_val_epoch = i_epoch

            if '_' not in opt.model_name:
                if not os.path.exists(
                        "{}/{}/{}".format(opt.save_model_dir, opt.model_name, opt.dataset)):
                    os.makedirs("{}/{}/{}".format(opt.save_model_dir,
                                opt.model_name, opt.dataset))
                path = "{0}/{1}/{2}/acc_{3}_f1_{4}_{5}.model".format(opt.save_model_dir, opt.model_name,
                                                                     opt.dataset,
                                                                     round(max_val_acc, 4), round(
                                                                         max_val_f1, 4),
                                                                     strftime("%y%m%d-%H%M", localtime()))
                torch.save(model.state_dict(), path)
                logger.info(">> saved: {}".format(path))
            elif 'lora' in opt.model_name:
                path = "peft/{0}/{1}//acc_{2}_f1_{3}_{4}".format(opt.model_name, opt.dataset, round(
                    max_val_acc, 4), round(max_val_f1, 4), strftime("%y%m%d-%H%M", localtime()))
                if not os.path.exists(path):
                    os.makedirs(path)
                model.save_pretrained(path)
                logger.info(">> saved: {}".format(path))
        if i_epoch - max_val_epoch >= opt.patience:
            print(">> early stop.")
            break
    return path


def evaluate_acc_f1(model, inputs_cols, data_loader):
    n_correct, n_total = 0, 0
    t_targets_all, t_logits_all = None, None
    # switch model to evaluation mode
    model.eval()
    with torch.no_grad():
        for i_batch, t_batch in enumerate(data_loader):
            t_inputs = {}
            for col in inputs_cols:
                t_inputs[col] = t_batch[col].to(opt.device)
            t_targets = t_batch["polarity"].to(opt.device)
            t_outputs = model(**t_inputs)
            t_logits = t_outputs.logits

            n_correct += (torch.argmax(t_logits, -1) == t_targets).sum().item()
            n_total += len(t_logits)

            if t_targets_all is None:
                t_targets_all = t_targets
                t_logits_all = t_logits
            else:
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                t_logits_all = torch.cat((t_logits_all, t_logits), dim=0)

    acc = n_correct / n_total
    f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_logits_all, -1).cpu(), labels=[0, 1, 2],
                          average="macro")
    return acc, f1


acc_list, f1_list = [], []
for i in range(opt.repeat):
    if opt.model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained("./plm/bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "./plm/bert-base-uncased", num_labels=3).to(opt.device)
        inputs_cols = ["input_ids", "token_type_ids", "attention_mask"]
    elif opt.model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("./plm/roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "./plm/roberta-base", num_labels=3).to(opt.device)
        inputs_cols = ["input_ids", "attention_mask"]
    elif opt.model_name == 'bert_lora':
        tokenizer = BertTokenizer.from_pretrained("./plm/bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "./plm/bert-base-uncased", num_labels=3).to(opt.device)
        base_model = model
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                 inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        inputs_cols = ["input_ids", "token_type_ids", "attention_mask"]
    elif opt.model_name == 'roberta_lora':
        tokenizer = RobertaTokenizer.from_pretrained("./plm/roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "./plm/roberta-base", num_labels=3).to(opt.device)
        base_model = model
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                 inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        inputs_cols = ["input_ids", "attention_mask"]

    # Dataloader
    trainset = ABSADataset(
        dataset_files[opt.dataset]["train"], tokenizer, opt.max_seq_len)
    testset = ABSADataset(
        dataset_files[opt.dataset]["test"], tokenizer, opt.max_seq_len)
    if opt.dataset == "mams":
        valset = ABSADataset(
            dataset_files[opt.dataset]["dev"], tokenizer, opt.max_seq_len)
    else:
        valset = testset
    train_data_loader = DataLoader(
        dataset=trainset, batch_size=opt.batch_size, shuffle=True)
    val_data_loader = DataLoader(
        dataset=valset, batch_size=opt.batch_size, shuffle=False)
    test_data_loader = DataLoader(
        dataset=testset, batch_size=opt.batch_size, shuffle=False)

    logger.info("cuda memory allocated: {}".format(
        torch.cuda.memory_allocated(device=opt.device.index)))
    print_args(model)

    # Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    _params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(_params, lr=opt.lr, weight_decay=opt.l2reg)

    best_model_path = train(model, inputs_cols, criterion,
                            optimizer, train_data_loader, val_data_loader)
    if '_' not in opt.model_name:
        model.load_state_dict(torch.load(best_model_path))
    elif 'lora' in opt.model_name:
        peft_model = best_model_path
        model = PeftModel.from_pretrained(
            base_model, peft_model, config=peft_config)
    test_acc, test_f1 = evaluate_acc_f1(model, inputs_cols, test_data_loader)
    logger.info(">> test_acc: {:.4f}, test_f1: {:.4f}".format(
        test_acc, test_f1))
    acc_list.append(test_acc)
    f1_list.append(test_f1)

all_acc = np.asarray(acc_list)
avg_acc = np.average(all_acc)
max_acc = np.max(all_acc)
all_f1 = np.asarray(f1_list)
avg_f1 = np.average(all_f1)
max_f1 = np.max(all_f1)
for acc, f1 in zip(acc_list, f1_list):
    logger.info(">> test_acc: {:.4f}, test_f1: {:.4f}".format(acc, f1))
logger.info(
    "\n>> avg_test_acc: {:.4f}, avg_test_f1: {:.4f}".format(avg_acc, avg_f1))
logger.info(
    ">> max_test_acc: {:.4f}, max_test_f1: {:.4f}".format(max_acc, max_f1))