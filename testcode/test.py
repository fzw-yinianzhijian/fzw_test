from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = "Oh, it is not my fault."
# inputs = tokenizer(text, return_tensors="pt")
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
print(inputs)

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

# from transformers import AutoModel

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModel.from_pretrained(checkpoint)

# outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)

# print(outputs.logits.shape)
print(outputs.logits)

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)