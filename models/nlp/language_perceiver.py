from transformers import PerceiverTokenizer, PerceiverForMaskedLM

tokenizer = PerceiverTokenizer.from_pretrained("deepmind/language-perceiver")
model = PerceiverForMaskedLM.from_pretrained("deepmind/language-perceiver")

text = "hello world"
inputs = tokenizer(text, return_tensors="pt").input_ids
outputs = model(inputs=inputs)
logits = outputs.logits
list(logits.shape)
