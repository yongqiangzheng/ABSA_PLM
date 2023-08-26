import torch.nn as nn


class RoBERTa(nn.Module):
    def __init__(self, roberta, args):
        super(RoBERTa, self).__init__()
        self.roberta = roberta
        self.dropout = nn.Dropout(args.dropout)
        self.dense = nn.Linear(args.bert_dim, args.polarities_dim)

    def forward(self, inputs):
        input_ids, attention_mask = inputs
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.pooler_output
        hidden_output = self.dropout(hidden)
        logits = self.dense(hidden_output)
        return logits
