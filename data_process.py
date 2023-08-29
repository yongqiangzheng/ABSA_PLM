import os
import re
import xml.etree.ElementTree as ET

from tqdm import tqdm

sentiment_dict = {"negative": "-1", "neutral": "0", "positive": "1"}


def xml2txt(path, domain, mode):
    sentence_list, aspect_list, label_list, aspect_index_list, sentence_id_list = [], [], [], [], []
    tree = ET.parse(path)  # 解析树
    root = tree.getroot()  # 根节点
    sent_id = 0
    for sentence in tqdm(root.findall("sentence")):
        aspectTerms = sentence.find("aspectTerms")
        if aspectTerms is None:  # 去掉没有aspect的句子
            continue
        text = sentence.find("text").text  # 句子
        for aspectTerm in aspectTerms.findall("aspectTerm"):  # 遍历所有的aspect
            polarity = aspectTerm.get("polarity").strip()
            if polarity == "conflict":  # 去掉conflict情感的句子
                continue
            aspect = aspectTerm.get("term")
            start = aspectTerm.get("from")
            end = aspectTerm.get("to")
            assert text[int(start):int(end)] == aspect
            sentence_list.append(text)
            aspect_list.append(aspect)
            label_list.append(sentiment_dict[polarity])
            aspect_index_list.append([int(start), int(end)])
            sentence_id_list.append(sent_id)
        sent_id += 1
    # write data
    if not os.path.exists("./dataset/txt"):
        os.mkdir("./dataset/txt")
    fout = open("dataset/txt/{}_{}.txt".format(domain, mode), "w")
    for sentence, aspect, label, aspect_index in tqdm(zip(sentence_list, aspect_list, label_list, aspect_index_list),
                                                      total=len(sentence_list)):
        mask_sentence = sentence[:aspect_index[0]] + " $T$ " + sentence[aspect_index[1]:]
        new_sentence = re.sub(" {2,}", " ", mask_sentence).strip()
        new_aspect = re.sub(" {2,}", " ", aspect).strip()
        fout.write(new_sentence + "\n" + new_aspect + "\n" + label + "\n")
    fout.close()


xml2txt("dataset/xml/SemEval2014/Laptops_Train.xml", "lap14", "train")
xml2txt("dataset/xml/SemEval2014/Laptops_Test.xml", "lap14", "test")
xml2txt("dataset/xml/SemEval2014/Restaurants_Train.xml", "rest14", "train")
xml2txt("dataset/xml/SemEval2014/Restaurants_Test.xml", "rest14", "test")
xml2txt("dataset/xml/MAMS-ATSA/train.xml", "mams", "train")
xml2txt("dataset/xml/MAMS-ATSA/val.xml", "mams", "dev")
xml2txt("dataset/xml/MAMS-ATSA/test.xml", "mams", "test")
