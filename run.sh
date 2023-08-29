python finetune.py --model_name bert --dataset lap14 --lr 2e-5
python finetune.py --model_name roberta --dataset lap14 --lr 2e-5
python finetune.py --model_name bert_lora --dataset lap14 --lr 1e-3
python finetune.py --model_name roberta_lora --dataset lap14 --lr 1e-3

python finetune.py --model_name bert --dataset rest14 --lr 2e-5
python finetune.py --model_name roberta --dataset rest14 --lr 2e-5
python finetune.py --model_name bert_lora --dataset rest14 --lr 1e-3
python finetune.py --model_name roberta_lora --dataset rest14 --lr 1e-3

python finetune.py --model_name bert --dataset mams --lr 2e-5
python finetune.py --model_name roberta --dataset mams --lr 2e-5
python finetune.py --model_name bert_lora --dataset mams --lr 1e-3
python finetune.py --model_name roberta_lora --dataset mams --lr 1e-3