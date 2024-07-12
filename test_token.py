from transformers import BertTokenizer

# 加载预训练的 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 对文本进行编码
encoded_input = tokenizer("Hello, world!")
print(encoded_input)

