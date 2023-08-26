import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, bert, args):
        super(BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(args.dropout)
        self.dense = nn.Linear(args.bert_dim, args.polarities_dim)

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden = output.pooler_output
        hidden_output = self.dropout(hidden)
        logits = self.dense(hidden_output)
        return logits
