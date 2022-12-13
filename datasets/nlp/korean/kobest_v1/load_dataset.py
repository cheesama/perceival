from datasets import load_dataset

import dill

boolq_dataset = load_dataset('skt/kobest_v1', 'boolq')
copa_dataset = load_dataset('skt/kobest_v1', 'copa')
sentineg_dataset = load_dataset('skt/kobest_v1', 'sentineg')
hellaswag_dataset = load_dataset('skt/kobest_v1', 'hellaswag')
wic_dataset = load_dataset('skt/kobest_v1', 'wic')

with open('./boolq_dataset.dill', 'wb') as f:
    dill.dump(boolq_dataset, f)
with open('./copa_dataset.dill', 'wb') as f:
    dill.dump(copa_dataset, f)
with open('./sentineg_dataset.dill', 'wb') as f:
    dill.dump(sentineg_dataset, f)
with open('./hellaswag_dataset.dill', 'wb') as f:
    dill.dump(hellaswag_dataset, f)
with open('./wic_dataset.dill', 'wb') as f:
    dill.dump(wic_dataset, f)
