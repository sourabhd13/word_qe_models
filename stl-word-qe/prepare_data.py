import pandas as pd

TRAIN_PATH = "train/"
TRAIN_SOURCE_FILE = "train_multi_all.src"
TRAIN_SOURCE_TAGS_FILE = "train_multi_all.source_tags"
TRAIN_TARGET_FILE = "train_multi_all.mt"
TRAIN_TARGET_TAGS_FLE = "train_multi_all.tags"

DEV_PATH = "dev/"
DEV_SOURCE_FILE = "dev_multi_all.src"
DEV_SOURCE_TAGS_FILE = "dev_multi_all.source_tags"
DEV_TARGET_FILE = "dev_multi_all.mt"
DEV_TARGET_TAGS_FLE = "dev_multi_all.tags"


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
print(train_df.head())
