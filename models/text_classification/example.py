from transformers import PerceiverTokenizer, PerceiverForSequenceClassification

tokenizer = PerceiverTokenizer.from_pretrained("deepmind/language-perceiver") #uses raw bytes utf-8 encoding, can be applied to korean, Too!
model = PerceiverForSequenceClassification.from_pretrained("deepmind/language-perceiver")

text = "hello world"
inputs = tokenizer(text, return_tensors="pt").input_ids
outputs = model(inputs=inputs)
logits = outputs.logits
list(logits.shape)
