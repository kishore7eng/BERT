
from transformers import BertTokenizer, BertConfig, BertForTokenClassification



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)
model = BertForTokenClassification.from_pretrained('bert-base-uncased', config=config)


model.save_pretrained("/workspace/amit_pg/biobert_ft/model")
tokenizer.save_pretrained('/workspace/amit_pg/biobert_ft/tokenizer')
config.save_pretrained("/workspace/amit_pg/biobert_ft/config")
