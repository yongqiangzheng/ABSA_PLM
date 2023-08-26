import argparse
import gradio as gr
import torch

from model.bert import BERT
from transformers import BertModel, BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="bert", type=str)
parser.add_argument("--dropout", default=0.3, type=float)
parser.add_argument("--batch_size", default=16, type=int, help="try 16, 32, 64 for BERT models")
parser.add_argument("--bert_dim", default=768, type=int)
parser.add_argument("--plm_name", default="./plm/bert-base-uncased", type=str)
parser.add_argument("--max_seq_len", default=128, type=int)
parser.add_argument("--polarities_dim", default=3, type=int)
parser.add_argument("--device", default=None, type=str, help="e.g. cuda:0")
opt = parser.parse_args()

model_classes = {
    "bert": BERT,
}
input_colses = {
    "bert": ["input_ids", "token_type_ids", "attention_mask"]
}

opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
    if opt.device is None else torch.device(opt.device)

bert = BertModel.from_pretrained(opt.plm_name)
tokenizer = BertTokenizer.from_pretrained(opt.plm_name)
model = BERT(bert, opt)
model.load_state_dict(torch.load("/media/b115/Backup/NLP/bert/lap14/acc_0.7821_f1_0.7481_230826-1625.model"))
model.eval()

examples = [["Boot time is super fast, around anywhere from 35 seconds to 1 minute.", "Boot time"],
            ["Did not enjoy the new Windows 8 and touchscreen functions.", "touchscreen functions"]
            ]


def greet(text, aspect):
    tensor_dict = tokenizer(text.lower(), aspect.lower(), padding="max_length", max_length=opt.max_seq_len,
                            return_tensors="pt")
    input_ids = tensor_dict["input_ids"]
    token_type_ids = tensor_dict["token_type_ids"]
    attention_mask = tensor_dict["attention_mask"]
    output = model([input_ids, token_type_ids, attention_mask])
    label = torch.argmax(output, -1).tolist()
    sentiment_dict = {0: "negative", 1: "neutral", 2: "positive"}
    result = sentiment_dict[label[0]]
    return result


demo = gr.Interface(fn=greet, inputs=["text", "text"], outputs="text",
                    examples=examples)
demo.launch()
