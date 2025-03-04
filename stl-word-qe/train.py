from multiprocessing import cpu_count
import pandas as pd
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel
import torch
import datetime
import time

start_time = datetime.datetime.now().time().strftime('%H:%M:%S')

TRAIN_PATH = "data/train/"
TRAIN_SOURCE_FILE = "train_multi_all.src"
TRAIN_SOURCE_TAGS_FILE = "train_multi_all.source_tags"
TRAIN_TARGET_FILE = "train_multi_all.mt"
TRAIN_TARGET_TAGS_FLE = "train_multi_all.tags"

DEV_PATH = "data/dev/"
DEV_SOURCE_FILE = "dev_multi_all.src"
DEV_SOURCE_TAGS_FILE = "dev_multi_all.source_tags"
DEV_TARGET_FILE = "dev_multi_all.mt"
DEV_TARGET_TAGS_FLE = "dev_multi_all.tags"

TEST_PATH = "data/test/"
TEST_SOURCE_FILE = "test_multi_all.src"
TEST_TARGET_FILE = "test_multi_all.mt"

TEST_SOURCE_TAGS_FILE = "predictions_src.txt"
TEST_TARGET_TAGS_FILE = "predictions_mt.txt"
TEST_TARGET_GAPS_FILE = "predictions_gaps.txt"

DEV_SOURCE_TAGS_FILE_SUB = "dev_predictions_src.txt"
DEV_TARGET_TAGS_FILE_SUB = "dev_predictions_mt.txt"
DEV_TARGET_GAPS_FILE_SUB = "dev_predictions_gaps.txt"


SEED = 777
TEMP_DIRECTORY = "temp/data"
GOOGLE_DRIVE = False
DRIVE_FILE_ID = None
MODEL_TYPE = "xlmroberta"
MODEL_NAME = "xlm-roberta-large"

microtransquest_config = {
    'output_dir': 'outputs/',
    "best_model_dir": "outputs/best_model",
    'cache_dir': 'cache_dir/',

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 80,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 4,
    'eval_batch_size': 8,
    'num_train_epochs': 1000,
    'weight_decay': 0,
    'learning_rate': 2e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 300,
    'save_steps': 300,
    "no_cache": False,
    "no_save": False,
    "save_recent_only": True,
    'save_model_every_epoch': False,
    'n_fold': 1,
    'evaluate_during_training': True,
    "evaluate_during_training_silent": True,
    'evaluate_during_training_steps': 300,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    "save_best_model": True,
    'save_eval_checkpoints': True,
    'tensorboard_dir': None,
    "save_optimizer_and_scheduler": True,

    'regression': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    "multiprocessing_chunksize": 500,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "early_stopping_consider_epochs": False,

    "manual_seed": SEED,

    "add_tag": False,
    "tag": "_",

    "default_quality": "OK",

    "config": {},
    "local_rank": -1,
    "encoding": None,

    "source_column": "source",
    "target_column": "target",
    "source_tags_column": "source_tags",
    "target_tags_column": "target_tags",
}



li_train_src = [x[:-1] for x in open(TRAIN_PATH+TRAIN_SOURCE_FILE).readlines()]
li_train_mt = [x[:-1] for x in open(TRAIN_PATH+TRAIN_TARGET_FILE).readlines()]
li_train_src_tags = [x[:-1] for x in open(TRAIN_PATH+TRAIN_SOURCE_TAGS_FILE).readlines()]
li_train_trg_tags = [x[:-1] for x in open(TRAIN_PATH+TRAIN_TARGET_TAGS_FLE).readlines()]

li_dev_src = [x[:-1] for x in open(DEV_PATH+DEV_SOURCE_FILE).readlines()]
li_dev_mt = [x[:-1] for x in open(DEV_PATH+DEV_TARGET_FILE).readlines()]
li_dev_src_tags = [x[:-1] for x in open(DEV_PATH+DEV_SOURCE_TAGS_FILE).readlines()]
li_dev_trg_tags = [x[:-1] for x in open(DEV_PATH+DEV_TARGET_TAGS_FLE).readlines()]

train_df = pd.DataFrame(list(zip(li_train_src, li_train_mt, li_train_src_tags, li_train_trg_tags)),
               columns =['source', 'target', 'source_tags', 'target_tags'])

eval_df = pd.DataFrame(list(zip(li_dev_src, li_dev_mt, li_dev_src_tags, li_dev_trg_tags)),
               columns =['source', 'target', 'source_tags', 'target_tags'])

model = MicroTransQuestModel("xlmroberta", "xlm-roberta-large", labels=["OK", "BAD"], use_cuda=torch.cuda.is_available(), args=microtransquest_config)
model.train_model(train_df , eval_data=eval_df)

end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
total_time=(datetime.datetime.strptime(end_time,'%H:%M:%S') - datetime.datetime.strptime(start_time,'%H:%M:%S'))
print ("Total time taken: ",total_time)
torch.cuda.empty_cache()
