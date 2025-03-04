from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel
import torch
import csv
import pandas as pd
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
import codecs
import numpy as np
from sklearn.metrics import f1_score

best_model_path="outputs/best_model"
test_src_path = "test_ru_en/test.src"
test_tgt_path = "test_ru_en/test.mt"
test_src_tags_path = "test_ru_en/test.source_tags"
test_mt_tags_path = "test_ru_en/test.tags"
lp = "ru-en"
gold_path = "gold_ru_en.txt"
pred_path = "predictions_ru_en.txt"
result_path = "results_test_ru_en.txt"


with open(test_src_path, 'r') as f1, open(test_tgt_path, 'r') as f2:
    source_sents = f1.readlines()
    target_sents = f2.readlines()

with open(test_src_tags_path, 'r') as f1, open(test_mt_tags_path, 'r') as f2:
	gold_source_tags = f1.readlines()
	gold_target_tags = f2.readlines()
	
gold_tags = []
for line1, line2 in zip(gold_source_tags, gold_target_tags):
    gold_tags.append(line1.split(" ") + line2.split(" "))

test = pd.DataFrame({'original': source_sents, 'translation': target_sents})
test = test[['original', 'translation']]
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b'}).dropna()

model = MicroTransQuestModel("xlmroberta", best_model_path, use_cuda=False)#torch.cuda.is_available())
test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))
source_tags, target_tags = model.predict(test_sentence_pairs, split_on_space=True)


predictions = []
for pred1, pred2 in zip(source_tags, target_tags):
    preds = pred1 + pred2
    predictions.append(preds)
    
f1_scores = []
for y_true, y_pred in zip(gold_tags, predictions):
    assert len(y_true) == len(y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    f1_scores.append(f1)
final_fscore = sum(f1_scores) / len(f1_scores)

with open(result_path, 'w') as f:
    f.write('final_f1_score: ')
    f.write(str(final_fscore))

with open(pred_path, 'w') as f:
    for preds in predictions:
        f.write(" ".join(preds))
        f.write("\n")
