import os
from transformers import AutoTokenizer, AutoModel

# pretrained_model = "bert-base-uncased"
pretrained_model = "roberta-base"

# load pre-trained weights
if not os.path.exists("./plm"):
    os.mkdir("./plm")
if not os.path.exists("./plm/{}".format(pretrained_model)):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModel.from_pretrained(pretrained_model)
    tokenizer.save_pretrained("./plm/{}".format(pretrained_model))
    model.save_pretrained("./plm/{}".format(pretrained_model))
else:
    print(pretrained_model + " exist")
