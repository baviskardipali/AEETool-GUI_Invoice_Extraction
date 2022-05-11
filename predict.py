"""
Access this file from the FLASK folder.
This files predicts the outcome of the model. It consists of certain preprocessing steps which are fundamental in recieving clean and appropriate outcomes.  It converts the raw output model to a structured format.
"""
from transformers import RobertaTokenizer, DistilBertTokenizer, AlbertTokenizer
import warnings
import config
import dataset
import numpy as np
import os
import re
from tokenizers import BertWordPieceTokenizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '1'
warnings.filterwarnings("ignore")

tokenizer_distilbert = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased')
tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer_bert = BertWordPieceTokenizer("Resource/vocab.txt", lowercase=True)
tokenizer_albert = AlbertTokenizer.from_pretrained('albert-base-v2')

# In detail:-
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed


def get_tokens(tokenized_sentence, tags, tags_name, enc_tag):
    map_ = {}
    for i in enc_tag.classes_:
        map_[i] = []

    for i in range(len(tags_name)):
        if tags[i] != 16 and tokenized_sentence[i] != 101:
            map_[tags_name[i]].append(tokenized_sentence[i])
    return map_


def get_mapping_roberta(sentence, meta_file, model_bert):
    final_dict = {"P_BUY_G": [], "P_BUY_N": [], "P_INV_DATE": [],
                  "P_INV_NO": [], "P_SUPP_G": [], "P_SUPP_N": [], "P_GT_AMT": []}
    extracted_info = {"P_BUY_G": [], "P_BUY_N": [], "P_INV_DATE": [
    ], "P_INV_NO": [], "P_SUPP_G": [], "P_SUPP_N": [], "P_GT_AMT": []}
    x_test, n_tokens = dataset.create_test_input_from_text_roberta(sentence)
    pred_test = model_bert.predict(x_test)
    pred_tags = np.argmax(pred_test, 2)[0][:n_tokens]
    tokenized_sentence = x_test[0][0][:n_tokens]
    meta_data = meta_file
    enc_tag = meta_data["enc_tag"]
    num_tag = len(list(enc_tag.classes_))

    le_dict = dict(zip(enc_tag.transform(
        enc_tag.classes_), enc_tag.classes_))
    tags_name = [le_dict.get(_, '[pad]') for _ in pred_tags]
    map = get_tokens(tokenized_sentence, pred_tags, tags_name, enc_tag)
    main_list = ['B-SUPP_N', 'I-SUPP_N', 'B-INV_NO', 'B-INV_DT',
                 'B-SUPP_G', 'B-BUY_N', 'I-BUY_N', 'B-BUY_G', 'B-GT_AMT']
    for i in map:
        if i in main_list:
            print(i, "-->", tokenizer_roberta.decode(map[i]))
        if i == "B-BUY_G":
            extracted_info['P_BUY_G'].append(tokenizer_roberta.decode(map[i]))
        if i == "I-BUY_G":
            extracted_info['P_BUY_G'].append(tokenizer_roberta.decode(map[i]))
            extracted_info['P_BUY_G'][0] = ''.join(extracted_info['P_BUY_G'])
            extracted_info['P_BUY_G'][0] = extracted_info['P_BUY_G'][0].replace(
                " ", "")
            # print(extracted_info['P_BUY_G'][0][:15])
            final_dict['P_BUY_G'].append(extracted_info['P_BUY_G'][0][:15])

        if i == "B-BUY_N":
            extracted_info['P_BUY_N'].append(tokenizer_roberta.decode(map[i]))
        if i == "I-BUY_N":
            extracted_info['P_BUY_N'].append(tokenizer_roberta.decode(map[i]))
            extracted_info['P_BUY_N'][0] = ''.join(extracted_info['P_BUY_N'])
            extracted_info['P_BUY_N'][0] = re.sub(
                '[^a-zA-Z]+', '', extracted_info['P_BUY_N'][0])
            final_dict['P_BUY_N'].append(extracted_info['P_BUY_N'][0])

        if i == "B-INV_DT":
            extracted_info['P_INV_DATE'].append(
                tokenizer_roberta.decode(map[i]))
            # print(tokenizer_roberta.decode(map[i]))
        if i == "I-INV_DT":
            # print(tokenizer_roberta.decode(map[i]))
            extracted_info['P_INV_DATE'].append(
                tokenizer_roberta.decode(map[i]))
            extracted_info['P_INV_DATE'][0] = ''.join(
                extracted_info['P_INV_DATE'])
            extracted_info['P_INV_DATE'][0] = extracted_info['P_INV_DATE'][0].replace(
                " ", "")
            final_dict['P_INV_DATE'].append(extracted_info['P_INV_DATE'][0])
            # print((extracted_info['P_INV_DATE'][0]))

        if i == "B-INV_NO":
            extracted_info['P_INV_NO'].append(tokenizer_roberta.decode(map[i]))
        if i == "I-INV_NO":
            extracted_info['P_INV_NO'].append(tokenizer_roberta.decode(map[i]))
            extracted_info['P_INV_NO'][0] = ''.join(extracted_info['P_INV_NO'])
            extracted_info['P_INV_NO'][0] = extracted_info['P_INV_NO'][0].replace(
                " ", "")
            final_dict['P_INV_NO'].append(extracted_info['P_INV_NO'][0])

        if i == "B-SUPP_G":
            extracted_info['P_SUPP_G'].append(tokenizer_roberta.decode(map[i]))
        if i == "I-SUPP_G":
            extracted_info['P_SUPP_G'].append(tokenizer_roberta.decode(map[i]))
            extracted_info['P_SUPP_G'][0] = ''.join(extracted_info['P_SUPP_G'])
            extracted_info['P_SUPP_G'][0] = extracted_info['P_SUPP_G'][0].replace(
                " ", "")
            final_dict['P_SUPP_G'].append(extracted_info['P_SUPP_G'][0])

        if i == "B-SUPP_N":
            extracted_info['P_SUPP_N'].append(tokenizer_roberta.decode(map[i]))
        if i == "I-SUPP_N":
            extracted_info['P_SUPP_N'].append(tokenizer_roberta.decode(map[i]))
            extracted_info['P_SUPP_N'][0] = ''.join(extracted_info['P_SUPP_N'])
            extracted_info['P_SUPP_N'][0] = re.sub(
                '[^a-zA-Z]+', '', extracted_info['P_SUPP_N'][0])
            final_dict['P_SUPP_N'].append(extracted_info['P_SUPP_N'][0])

        if i == "B-GT_AMT":
            extracted_info['P_GT_AMT'].append(tokenizer_roberta.decode(map[i]))
        if i == "I-GT_AMT":
            extracted_info['P_GT_AMT'].append(tokenizer_roberta.decode(map[i]))
            extracted_info['P_GT_AMT'][0] = ''.join(extracted_info['P_GT_AMT'])
            extracted_info['P_GT_AMT'][0] = extracted_info['P_GT_AMT'][0].replace(
                " ", "")
            extracted_info['P_GT_AMT'][0] = extracted_info['P_GT_AMT'][0].replace(
                ',', "")
            extracted_info['P_GT_AMT'][0], sep, tail = extracted_info['P_GT_AMT'][0].partition(
                '.')
            extracted_info['P_GT_AMT'][0] = re.sub(
                "[^0-9]", "", extracted_info['P_GT_AMT'][0])
            if str(extracted_info['P_GT_AMT'][0][-2:]) == '00':
                extracted_info['P_GT_AMT'][0] = extracted_info['P_GT_AMT'][0][:-2]
            final_dict['P_GT_AMT'].append(str(extracted_info['P_GT_AMT'][0]))

    return final_dict


def get_mapping_bert(sentence, meta_file, model_bert):
    final_dict = {"P_BUY_G": [], "P_BUY_N": [], "P_INV_DATE": [],
                  "P_INV_NO": [], "P_SUPP_G": [], "P_SUPP_N": [], "P_GT_AMT": []}
    extracted_info = {"P_BUY_G": [], "P_BUY_N": [], "P_INV_DATE": [
    ], "P_INV_NO": [], "P_SUPP_G": [], "P_SUPP_N": [], "P_GT_AMT": []}
    x_test, n_tokens = dataset.create_test_input_from_text_bert(sentence)
    pred_test = model_bert.predict(x_test)
    pred_tags = np.argmax(pred_test, 2)[0][:n_tokens]
    tokenized_sentence = x_test[0][0][:n_tokens]
    meta_data = meta_file
    enc_tag = meta_data["enc_tag"]
    num_tag = len(list(enc_tag.classes_))

    le_dict = dict(zip(enc_tag.transform(
        enc_tag.classes_), enc_tag.classes_))
    tags_name = [le_dict.get(_, '[pad]') for _ in pred_tags]
    map = get_tokens(tokenized_sentence, pred_tags, tags_name, enc_tag)
    main_list = ['B-SUPP_N', 'I-SUPP_N', 'B-INV_NO', 'B-INV_DT',
                 'B-SUPP_G', 'B-BUY_N', 'I-BUY_N', 'B-BUY_G', 'B-GT_AMT']
    for i in map:
        if i in main_list:
            print(i, "-->", tokenizer_bert.decode(map[i]))
        if i == "B-BUY_G":
            extracted_info['P_BUY_G'].append(tokenizer_bert.decode(map[i]))
        if i == "I-BUY_G":
            extracted_info['P_BUY_G'].append(tokenizer_bert.decode(map[i]))
            extracted_info['P_BUY_G'][0] = ''.join(extracted_info['P_BUY_G'])
            extracted_info['P_BUY_G'][0] = extracted_info['P_BUY_G'][0].replace(
                " ", "")
            # print(extracted_info['P_BUY_G'][0][:15])
            final_dict['P_BUY_G'].append(extracted_info['P_BUY_G'][0][:15])

        if i == "B-BUY_N":
            extracted_info['P_BUY_N'].append(tokenizer_bert.decode(map[i]))
        if i == "I-BUY_N":
            extracted_info['P_BUY_N'].append(tokenizer_bert.decode(map[i]))
            extracted_info['P_BUY_N'][0] = ''.join(extracted_info['P_BUY_N'])
            extracted_info['P_BUY_N'][0] = re.sub(
                '[^a-zA-Z]+', '', extracted_info['P_BUY_N'][0])
            final_dict['P_BUY_N'].append(extracted_info['P_BUY_N'][0])

        if i == "B-INV_DT":
            extracted_info['P_INV_DATE'].append(tokenizer_bert.decode(map[i]))
            # print(tokenizer_bert.decode(map[i]))
        if i == "I-INV_DT":
            # print(tokenizer_bert.decode(map[i]))
            extracted_info['P_INV_DATE'].append(tokenizer_bert.decode(map[i]))
            extracted_info['P_INV_DATE'][0] = ''.join(
                extracted_info['P_INV_DATE'])
            extracted_info['P_INV_DATE'][0] = extracted_info['P_INV_DATE'][0].replace(
                " ", "")
            final_dict['P_INV_DATE'].append(extracted_info['P_INV_DATE'][0])
            # print((extracted_info['P_INV_DATE'][0]))

        if i == "B-INV_NO":
            extracted_info['P_INV_NO'].append(tokenizer_bert.decode(map[i]))
        if i == "I-INV_NO":
            extracted_info['P_INV_NO'].append(tokenizer_bert.decode(map[i]))
            extracted_info['P_INV_NO'][0] = ''.join(extracted_info['P_INV_NO'])
            extracted_info['P_INV_NO'][0] = extracted_info['P_INV_NO'][0].replace(
                " ", "")
            final_dict['P_INV_NO'].append(extracted_info['P_INV_NO'][0])

        if i == "B-SUPP_G":
            extracted_info['P_SUPP_G'].append(tokenizer_bert.decode(map[i]))
        if i == "I-SUPP_G":
            extracted_info['P_SUPP_G'].append(tokenizer_bert.decode(map[i]))
            extracted_info['P_SUPP_G'][0] = ''.join(extracted_info['P_SUPP_G'])
            extracted_info['P_SUPP_G'][0] = extracted_info['P_SUPP_G'][0].replace(
                " ", "")
            final_dict['P_SUPP_G'].append(extracted_info['P_SUPP_G'][0])

        if i == "B-SUPP_N":
            extracted_info['P_SUPP_N'].append(tokenizer_bert.decode(map[i]))
        if i == "I-SUPP_N":
            extracted_info['P_SUPP_N'].append(tokenizer_bert.decode(map[i]))
            extracted_info['P_SUPP_N'][0] = ''.join(extracted_info['P_SUPP_N'])
            extracted_info['P_SUPP_N'][0] = re.sub(
                '[^a-zA-Z]+', '', extracted_info['P_SUPP_N'][0])
            final_dict['P_SUPP_N'].append(extracted_info['P_SUPP_N'][0])

        if i == "B-GT_AMT":
            extracted_info['P_GT_AMT'].append(tokenizer_bert.decode(map[i]))
        if i == "I-GT_AMT":
            extracted_info['P_GT_AMT'].append(tokenizer_bert.decode(map[i]))
            extracted_info['P_GT_AMT'][0] = ''.join(extracted_info['P_GT_AMT'])
            extracted_info['P_GT_AMT'][0] = extracted_info['P_GT_AMT'][0].replace(
                " ", "")
            extracted_info['P_GT_AMT'][0] = extracted_info['P_GT_AMT'][0].replace(
                ',', "")
            extracted_info['P_GT_AMT'][0], sep, tail = extracted_info['P_GT_AMT'][0].partition(
                '.')
            extracted_info['P_GT_AMT'][0] = re.sub(
                "[^0-9]", "", extracted_info['P_GT_AMT'][0])
            if str(extracted_info['P_GT_AMT'][0][-2:]) == '00':
                extracted_info['P_GT_AMT'][0] = extracted_info['P_GT_AMT'][0][:-2]
            final_dict['P_GT_AMT'].append(str(extracted_info['P_GT_AMT'][0]))

    return final_dict


def get_mapping_distilbert(sentence, meta_file, model_bert):
    final_dict = {"P_BUY_G": [], "P_BUY_N": [], "P_INV_DATE": [],
                  "P_INV_NO": [], "P_SUPP_G": [], "P_SUPP_N": [], "P_GT_AMT": []}
    extracted_info = {"P_BUY_G": [], "P_BUY_N": [], "P_INV_DATE": [
    ], "P_INV_NO": [], "P_SUPP_G": [], "P_SUPP_N": [], "P_GT_AMT": []}
    x_test, n_tokens = dataset.create_test_input_from_text_distilbert(sentence)
    pred_test = model_bert.predict(x_test)
    pred_tags = np.argmax(pred_test, 2)[0][:n_tokens]
    tokenized_sentence = x_test[0][0][:n_tokens]
    meta_data = meta_file
    enc_tag = meta_data["enc_tag"]
    num_tag = len(list(enc_tag.classes_))

    le_dict = dict(zip(enc_tag.transform(
        enc_tag.classes_), enc_tag.classes_))
    tags_name = [le_dict.get(_, '[pad]') for _ in pred_tags]
    map = get_tokens(tokenized_sentence, pred_tags, tags_name, enc_tag)
    main_list = ['B-SUPP_N', 'I-SUPP_N', 'B-INV_NO', 'B-INV_DT',
                 'B-SUPP_G', 'B-BUY_N', 'I-BUY_N', 'B-BUY_G', 'B-GT_AMT']
    for i in map:
        if i in main_list:
            print(i, "-->", tokenizer_distilbert.decode(map[i]))
        if i == "B-BUY_G":
            extracted_info['P_BUY_G'].append(
                tokenizer_distilbert.decode(map[i]))
        if i == "I-BUY_G":
            extracted_info['P_BUY_G'].append(
                tokenizer_distilbert.decode(map[i]))
            extracted_info['P_BUY_G'][0] = ''.join(extracted_info['P_BUY_G'])
            extracted_info['P_BUY_G'][0] = extracted_info['P_BUY_G'][0].replace(
                " ", "")
            # print(extracted_info['P_BUY_G'][0][:15])
            final_dict['P_BUY_G'].append(extracted_info['P_BUY_G'][0][:15])

        if i == "B-BUY_N":
            extracted_info['P_BUY_N'].append(
                tokenizer_distilbert.decode(map[i]))
        if i == "I-BUY_N":
            extracted_info['P_BUY_N'].append(
                tokenizer_distilbert.decode(map[i]))
            extracted_info['P_BUY_N'][0] = ''.join(extracted_info['P_BUY_N'])
            extracted_info['P_BUY_N'][0] = re.sub(
                '[^a-zA-Z]+', '', extracted_info['P_BUY_N'][0])
            final_dict['P_BUY_N'].append(extracted_info['P_BUY_N'][0])

        if i == "B-INV_DT":
            extracted_info['P_INV_DATE'].append(
                tokenizer_distilbert.decode(map[i]))
            # print(tokenizer_distilbert.decode(map[i]))
        if i == "I-INV_DT":
            # print(tokenizer_distilbert.decode(map[i]))
            extracted_info['P_INV_DATE'].append(
                tokenizer_distilbert.decode(map[i]))
            extracted_info['P_INV_DATE'][0] = ''.join(
                extracted_info['P_INV_DATE'])
            extracted_info['P_INV_DATE'][0] = extracted_info['P_INV_DATE'][0].replace(
                " ", "")
            final_dict['P_INV_DATE'].append(extracted_info['P_INV_DATE'][0])
            # print((extracted_info['P_INV_DATE'][0]))

        if i == "B-INV_NO":
            extracted_info['P_INV_NO'].append(
                tokenizer_distilbert.decode(map[i]))
        if i == "I-INV_NO":
            extracted_info['P_INV_NO'].append(
                tokenizer_distilbert.decode(map[i]))
            extracted_info['P_INV_NO'][0] = ''.join(extracted_info['P_INV_NO'])
            extracted_info['P_INV_NO'][0] = extracted_info['P_INV_NO'][0].replace(
                " ", "")
            final_dict['P_INV_NO'].append(extracted_info['P_INV_NO'][0])

        if i == "B-SUPP_G":
            extracted_info['P_SUPP_G'].append(
                tokenizer_distilbert.decode(map[i]))
        if i == "I-SUPP_G":
            extracted_info['P_SUPP_G'].append(
                tokenizer_distilbert.decode(map[i]))
            extracted_info['P_SUPP_G'][0] = ''.join(extracted_info['P_SUPP_G'])
            extracted_info['P_SUPP_G'][0] = extracted_info['P_SUPP_G'][0].replace(
                " ", "")
            final_dict['P_SUPP_G'].append(extracted_info['P_SUPP_G'][0])

        if i == "B-SUPP_N":
            extracted_info['P_SUPP_N'].append(
                tokenizer_distilbert.decode(map[i]))
        if i == "I-SUPP_N":
            extracted_info['P_SUPP_N'].append(
                tokenizer_distilbert.decode(map[i]))
            extracted_info['P_SUPP_N'][0] = ''.join(extracted_info['P_SUPP_N'])
            extracted_info['P_SUPP_N'][0] = re.sub(
                '[^a-zA-Z]+', '', extracted_info['P_SUPP_N'][0])
            final_dict['P_SUPP_N'].append(extracted_info['P_SUPP_N'][0])

        if i == "B-GT_AMT":
            extracted_info['P_GT_AMT'].append(
                tokenizer_distilbert.decode(map[i]))
        if i == "I-GT_AMT":
            extracted_info['P_GT_AMT'].append(
                tokenizer_distilbert.decode(map[i]))
            extracted_info['P_GT_AMT'][0] = ''.join(extracted_info['P_GT_AMT'])
            extracted_info['P_GT_AMT'][0] = extracted_info['P_GT_AMT'][0].replace(
                " ", "")
            extracted_info['P_GT_AMT'][0] = extracted_info['P_GT_AMT'][0].replace(
                ',', "")
            extracted_info['P_GT_AMT'][0], sep, tail = extracted_info['P_GT_AMT'][0].partition(
                '.')
            extracted_info['P_GT_AMT'][0] = re.sub(
                "[^0-9]", "", extracted_info['P_GT_AMT'][0])
            if str(extracted_info['P_GT_AMT'][0][-2:]) == '00':
                extracted_info['P_GT_AMT'][0] = extracted_info['P_GT_AMT'][0][:-2]
            final_dict['P_GT_AMT'].append(str(extracted_info['P_GT_AMT'][0]))

    return final_dict

def get_mapping_albert(sentence, meta_file, model_bert):
  final_dict = {"P_BUY_G": [], "P_BUY_N": [], "P_INV_DATE": [],
                  "P_INV_NO": [], "P_SUPP_G": [], "P_SUPP_N": [], "P_GT_AMT": []}
  extracted_info = {"P_BUY_G": [], "P_BUY_N": [], "P_INV_DATE": [], "P_INV_NO": [], "P_SUPP_G": [], "P_SUPP_N": [], "P_GT_AMT": []}
  test_inputs = [sentence]
  # test_inputs = ["alex lives in london"]
  x_test, n_tokens = dataset.create_test_input_from_text_albert(sentence)
  pred_test = model_bert.predict(x_test)
  pred_tags = np.argmax(pred_test,2)[0][:n_tokens]  # ignore predictions of padding tokens

  meta_data = meta_file
  tag_encoder = meta_data["enc_tag"]
  num_tag = len(list(tag_encoder.classes_))

  # create dictionary of tag and its index
  le_dict = dict(zip(tag_encoder.transform(tag_encoder.classes_), tag_encoder.classes_))
  tokenized_sentence = x_test[0][0][:n_tokens]
  tags_name = [le_dict.get(_, '[pad]') for _ in pred_tags]
  map = get_tokens(tokenized_sentence, pred_tags, tags_name, tag_encoder)
  main_list = ['B-SUPP_N','I-SUPP_N','B-INV_NO','B-INV_DT','B-SUPP_G','B-BUY_N','I-BUY_N','B-BUY_G','B-GT_AMT']
  for i in map:
    if i in main_list:
      print(i,"-->",tokenizer_albert.decode(map[i]))
    if i == "B-BUY_G":
      extracted_info['P_BUY_G'].append(tokenizer_albert.decode(map[i]))
    if i == "I-BUY_G":
      extracted_info['P_BUY_G'].append(tokenizer_albert.decode(map[i]))
      extracted_info['P_BUY_G'][0] = ''.join(extracted_info['P_BUY_G'])
      extracted_info['P_BUY_G'][0] = extracted_info['P_BUY_G'][0].replace(" ", "")
      #print(extracted_info['P_BUY_G'][0][:15])
      final_dict['P_BUY_G'].append(extracted_info['P_BUY_G'][0][:15])

    if i == "B-BUY_N":
      extracted_info['P_BUY_N'].append(tokenizer_albert.decode(map[i]))
    if i == "I-BUY_N":
      extracted_info['P_BUY_N'].append(tokenizer_albert.decode(map[i]))
      extracted_info['P_BUY_N'][0] = ''.join(extracted_info['P_BUY_N'])
      extracted_info['P_BUY_N'][0] = re.sub('[^a-zA-Z]+', '', extracted_info['P_BUY_N'][0])
      final_dict['P_BUY_N'].append(extracted_info['P_BUY_N'][0])

    if i == "B-INV_DT":
      extracted_info['P_INV_DATE'].append(tokenizer_albert.decode(map[i]))
      # print(tokenizer.decode(map[i]))
    if i == "I-INV_DT":
      # print(tokenizer.decode(map[i]))
      extracted_info['P_INV_DATE'].append(tokenizer_albert.decode(map[i]))
      extracted_info['P_INV_DATE'][0] = ''.join(extracted_info['P_INV_DATE'])
      extracted_info['P_INV_DATE'][0] = extracted_info['P_INV_DATE'][0].replace(" ", "")
      final_dict['P_INV_DATE'].append(extracted_info['P_INV_DATE'][0])
      # print((extracted_info['P_INV_DATE'][0]))

    if i == "B-INV_NO":
      extracted_info['P_INV_NO'].append(tokenizer_albert.decode(map[i]))
    if i == "I-INV_NO":
      extracted_info['P_INV_NO'].append(tokenizer_albert.decode(map[i]))
      extracted_info['P_INV_NO'][0] = ''.join(extracted_info['P_INV_NO'])
      extracted_info['P_INV_NO'][0] = extracted_info['P_INV_NO'][0].replace(" ", "")
      final_dict['P_INV_NO'].append(extracted_info['P_INV_NO'][0])
    
    if i == "B-SUPP_G":
      extracted_info['P_SUPP_G'].append(tokenizer_albert.decode(map[i]))
    if i == "I-SUPP_G":
      extracted_info['P_SUPP_G'].append(tokenizer_albert.decode(map[i]))
      extracted_info['P_SUPP_G'][0] = ''.join(extracted_info['P_SUPP_G'])
      extracted_info['P_SUPP_G'][0] = extracted_info['P_SUPP_G'][0].replace(" ", "")
      final_dict['P_SUPP_G'].append(extracted_info['P_SUPP_G'][0])

    if i == "B-SUPP_N":
      extracted_info['P_SUPP_N'].append(tokenizer_albert.decode(map[i]))
    if i == "I-SUPP_N":
      extracted_info['P_SUPP_N'].append(tokenizer_albert.decode(map[i]))
      extracted_info['P_SUPP_N'][0] = ''.join(extracted_info['P_SUPP_N'])
      extracted_info['P_SUPP_N'][0] = re.sub('[^a-zA-Z]+', '', extracted_info['P_SUPP_N'][0])
      final_dict['P_SUPP_N'].append(extracted_info['P_SUPP_N'][0])

    if i == "B-GT_AMT":
      extracted_info['P_GT_AMT'].append(tokenizer_albert.decode(map[i]))
    if i == "I-GT_AMT":
      extracted_info['P_GT_AMT'].append(tokenizer_albert.decode(map[i]))
      extracted_info['P_GT_AMT'][0] = ''.join(extracted_info['P_GT_AMT'])
      extracted_info['P_GT_AMT'][0] = extracted_info['P_GT_AMT'][0].replace(" ", "")
      extracted_info['P_GT_AMT'][0] = extracted_info['P_GT_AMT'][0].replace(',', "")
      extracted_info['P_GT_AMT'][0], sep, tail = extracted_info['P_GT_AMT'][0].partition('.')
      extracted_info['P_GT_AMT'][0] = re.sub("[^0-9]", "", extracted_info['P_GT_AMT'][0])
      if str(extracted_info['P_GT_AMT'][0][-2:]) == '00':
          extracted_info['P_GT_AMT'][0] =  extracted_info['P_GT_AMT'][0][:-2]
      final_dict['P_GT_AMT'].append(str(extracted_info['P_GT_AMT'][0]))
  return final_dict


# if __name__ == "__main__":
#     meta_data = joblib.load("meta.bin")
#     enc_tag = meta_data["enc_tag"]
#     num_tag = len(list(enc_tag.classes_))
#     m1 = create_model(num_tag)
#     m1.load_weights(config.WEIGHT_PATH)

#     TEXT_FILE_PATH = r'Text/*.txt'
#     var = glob.glob(TEXT_FILE_PATH)
#     var.sort()
#     for i in var:
#         f = open(i, "r")

#         sentence = f.read()

#         get_mapping([sentence])

#     extracted_df = pd.DataFrame(final_info)
#     print(extracted_df)
#     extracted_df.to_csv(config.EXTRACTED_FILE, header=True, index=False)
