import os
import re
import xml.etree.ElementTree as ET

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# load pre-trained weights
if not os.path.exists('./plm'):
    os.mkdir('./plm')
if not os.path.exists('./plm/bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained('./plm/bert-base-uncased')
    model.save_pretrained('./plm/bert-base-uncased')
else:
    tokenizer = AutoTokenizer.from_pretrained('./plm/bert-base-uncased')
    model = AutoModel.from_pretrained('./plm/bert-base-uncased')

sentiment_dict = {'negative': '-1', 'neutral': '0', 'positive': '1'}


def parse_semeval14(path, domain, mode):
    sentence_list, aspect_list, label_list, aspect_index_list, sentence_id_list = [], [], [], [], []
    tree = ET.parse(path)  # 解析树
    root = tree.getroot()  # 根节点
    sent_id = 0
    for sentence in tqdm(root.findall('sentence')):
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:  # 去掉没有aspect的句子
            continue
        text = sentence.find('text').text  # 句子
        for aspectTerm in aspectTerms.findall('aspectTerm'):  # 遍历所有的aspect
            polarity = aspectTerm.get('polarity').strip()
            if polarity == 'conflict':  # 去掉conflict情感的句子
                continue
            aspect = aspectTerm.get('term')
            start = aspectTerm.get('from')
            end = aspectTerm.get('to')
            assert text[int(start):int(end)] == aspect
            sentence_list.append(text)
            aspect_list.append(aspect)
            label_list.append(sentiment_dict[polarity])
            aspect_index_list.append([int(start), int(end)])
            sentence_id_list.append(sent_id)
        sent_id += 1
    # write data
    if not os.path.exists('./datasets/txt'):
        os.mkdir('./datasets/txt')
    fout = open('datasets/txt/{}_{}.txt'.format(domain, mode), 'w')
    for sentence, aspect, label, aspect_index in tqdm(zip(sentence_list, aspect_list, label_list, aspect_index_list),
                                                      total=len(sentence_list)):
        mask_sentence = sentence[:aspect_index[0]] + ' $T$ ' + sentence[aspect_index[1]:]
        new_sentence = re.sub(' {2,}', ' ', mask_sentence).strip()
        new_aspect = re.sub(' {2,}', ' ', aspect).strip()

        fout.write(new_sentence + '\n' + new_aspect + '\n' + label + '\n')
    fout.close()
    # write tensor
    if not os.path.exists('./datasets/input'):
        os.mkdir('./datasets/input')
    tensor_dict = tokenizer(sentence_list, aspect_list, padding='longest')
    tensor_dict['label'] = [int(l) for l in label_list]
    torch.save(tensor_dict, 'datasets/input/{}_{}.pt'.format(domain, mode))


parse_semeval14('datasets/xml/SemEval2014/Laptops_Train.xml', 'lap14', 'train')
parse_semeval14('datasets/xml/SemEval2014/Laptops_Test.xml', 'lap14', 'test')
parse_semeval14('datasets/xml/SemEval2014/Restaurants_Train.xml', 'rest14', 'train')
parse_semeval14('datasets/xml/SemEval2014/Restaurants_Test.xml', 'rest14', 'test')

# def parse_semeval1516(path, domain, mode):
#     sentence_list, aspect_list, label_list, aspect_index_list, sentence_id_list = [], [], [], [], []
#     tree = ET.parse(path)  # 解析树
#     root = tree.getroot()  # 根节点
#     sent_id = 0
#     for review in tqdm(root.findall('Review')):
#         for sentences in review.findall('sentences'):
#             for sentence in sentences.findall('sentence'):
#                 text = sentence.find('text').text  # 句子
#                 if not sentence.findall('Opinions'):  # 删除没有aspect的句子
#                     continue
#                 for opinions in sentence.findall('Opinions'):
#                     for opinion in opinions.findall('Opinion'):  # 遍历所有的aspect
#                         aspect = opinion.get('target')
#                         if aspect == 'NULL':
#                             continue
#                         polarity = opinion.get('polarity').strip()
#                         start = opinion.get('from')
#                         end = opinion.get('to')
#                         assert text[int(start):int(end)] == aspect
#                         sentence_list.append(text)
#                         aspect_list.append(aspect)
#                         label_list.append(sentiment_dict[polarity])
#                         aspect_index_list.append([int(start), int(end)])
#                         sentence_id_list.append(sent_id)
#                 sent_id += 1
#     # write data
#     if not os.path.exists('./datasets/txt'):
#         os.mkdir('./datasets/txt')
#     fout = open('datasets/txt/{}_{}.txt'.format(domain, mode), 'w')
#     for sentence, aspect, label, aspect_index in tqdm(zip(sentence_list, aspect_list, label_list, aspect_index_list),
#                                                       total=len(sentence_list)):
#         mask_sentence = sentence[:aspect_index[0]] + ' $T$ ' + sentence[aspect_index[1]:]
#         new_sentence = re.sub(' {2,}', ' ', mask_sentence).strip()
#         new_aspect = re.sub(' {2,}', ' ', aspect).strip()
#         fout.write(new_sentence + '\n' + new_aspect + '\n' + label + '\n')
#     fout.close()
#     # write tensor
#     if not os.path.exists('./datasets/input'):
#         os.mkdir('./datasets/input')
#     tensor_dict = tokenizer(sentence_list, aspect_list, padding='longest')
#     tensor_dict['label'] = [int(l) for l in label_list]
#     torch.save(tensor_dict, 'datasets/input/{}_{}.pt'.format(domain, mode))

# parse_semeval1516('./datasets/xml/SemEval2015/Restaurants_Train.xml', 'rest15', 'train')
# parse_semeval1516('./datasets/xml/SemEval2015/Restaurants_Test.xml', 'rest15', 'test')
# parse_semeval1516('./datasets/xml/SemEval2016/Restaurants_Train.xml', 'rest16', 'train')
# parse_semeval1516('./datasets/xml/SemEval2016/Restaurants_Test.xml', 'rest16', 'test')
