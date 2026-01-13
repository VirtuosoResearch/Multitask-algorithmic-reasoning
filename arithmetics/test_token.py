from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_input = tokenizer("Hello, world!")
print(encoded_input)
