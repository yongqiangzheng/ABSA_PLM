import argparse
import gradio as gr
import torch

from sklearn import metrics
from transformers import BertTokenizer, RobertaTokenizer, BertForSequenceClassification, RobertaForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader, Dataset


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


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="roberta_lora", type=str)
parser.add_argument("--dataset", default="lap14", type=str,
                    help="lap14, rest14")
parser.add_argument("--dropout", default=0.3, type=float)
parser.add_argument("--batch_size", default=16, type=int,
                    help="try 16, 32, 64 for BERT models")
parser.add_argument("--bert_dim", default=768, type=int)
parser.add_argument("--plm_name", default="./plm/bert-base-uncased", type=str)
parser.add_argument("--max_seq_len", default=128, type=int)
parser.add_argument("--polarities_dim", default=3, type=int)
parser.add_argument("--device", default="cuda", type=str, help="e.g. cuda:0")
opt = parser.parse_args()

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

if opt.model_name == 'bert':
    tokenizer = BertTokenizer.from_pretrained("./plm/bert-base-uncased")
    ft_model = '/media/b115/NLP/bert/acc_0.7884_f1_0.7501_230828-0342.model'
    model = BertForSequenceClassification.from_pretrained(
        ft_model, num_labels=3).to(opt.device)
    inputs_cols = ["input_ids", "token_type_ids", "attention_mask"]
elif opt.model_name == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained("./plm/roberta-base")
    ft_model = '/media/b115/NLP/roberta/acc_0.8197_f1_0.7839_230828-0411.model'
    model = RobertaForSequenceClassification.from_pretrained(
        ft_model, num_labels=3).to(opt.device)
    inputs_cols = ["input_ids", "attention_mask"]
elif opt.model_name == 'bert_lora':
    tokenizer = BertTokenizer.from_pretrained("./plm/bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "./plm/bert-base-uncased", num_labels=3).to(opt.device)
    base_model = model
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                             inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1)
    peft_model = './checkpoint/bert_lora_lap14_acc_0.7884_f1_0.7498_230828-0415'
    model = PeftModel.from_pretrained(
        base_model, peft_model, config=peft_config)
    inputs_cols = ["input_ids", "token_type_ids", "attention_mask"]
elif opt.model_name == 'roberta_lora':
    tokenizer = RobertaTokenizer.from_pretrained("./plm/roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "./plm/roberta-base", num_labels=3).to(opt.device)
    base_model = model
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                             inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1)
    peft_model = './checkpoint/roberta_lora_lap14_acc_0.8276_f1_0.7997_230828-0429'
    model = PeftModel.from_pretrained(
        base_model, peft_model, config=peft_config)
    inputs_cols = ["input_ids", "attention_mask"]

model.eval()

testset = ABSADataset(
    dataset_files[opt.dataset]["test"], tokenizer, opt.max_seq_len)
test_data_loader = DataLoader(
    dataset=testset, batch_size=opt.batch_size, shuffle=False)
test_acc, test_f1 = evaluate_acc_f1(model, inputs_cols, test_data_loader)
print(test_acc, test_f1)

examples = [["Boot time is super fast, around anywhere from 35 seconds to 1 minute.", "Boot time"],
            ["Did not enjoy the new Windows 8 and touchscreen functions.",
                "touchscreen functions"]
            ]


def greet(text, aspect):
    tensor_dict = tokenizer(text.lower(), aspect.lower(), padding="max_length", max_length=opt.max_seq_len,
                            return_tensors="pt")
    inputs = {}
    for col in inputs_cols:
        inputs[col] = tensor_dict[col].to(opt.device)      
    output = model(**inputs)
    logits = output.logits
    label = torch.argmax(logits, -1).tolist()
    sentiment_dict = {0: "negative", 1: "neutral", 2: "positive"}
    result = sentiment_dict[label[0]]
    return result


demo = gr.Interface(fn=greet, inputs=["text", "text"], outputs="text",
                    examples=examples)
demo.launch()
