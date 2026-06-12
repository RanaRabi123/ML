from transformers import BertTokenizerFast

# Load the pretrained BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

text = ["Analyzing microelectronics",
        "Artificial intelligence is transforming the modern world through automation and generation",
        "She walked down the quiet street to the old bakery",
        "Learning a new language requires dedication and daily practice",
        "The local market was bustling with people and fresh produce",
        "Renewable energy sources are becoming more efficient every year",
        "He carefully reviewed the document before signing it",
        "Yesterday's heavy rain caused flooding in several low-lying areas",
        "Cooking a delicious meal can be a great way to unwind",
        "The scientist published a groundbreaking paper on genetics",
        "We decided to take a scenic drive through the mountains",
        "Time management is crucial for achieving professional and personal goals",
        "They built a magnificent sandcastle on the bright, sunny beach",
        "Good communication is the foundation of any strong relationship",
        "The historical museum offers guided tours on weekends",
        "She plays the piano with incredible passion and precision",
        "The new software update improves overall system performance",
        "Traveling exposes us to different cultures and perspectives",
        "He ordered a cup of black coffee and a croissant",
        "The starry night sky stretched endlessly above the horizon"]

tokens = []
for sentence in text:
    tokens.append(tokenizer.tokenize(sentence))

input_ids = []
for i in tokens:
  input_ids.append(tokenizer.convert_tokens_to_ids(i))

print('tokens : ', tokens[0],'\n','Model Input id', input_ids[0])




            # ------------------------------
encoded = []
for i in text:
    encoded.append(tokenizer(i))

id_token = []
for i in range(len(encoded)):
   id_token.append(tokenizer.convert_ids_to_tokens(encoded[i]['input_ids']))

word_indexes=[]
for i in range(len(id_token)):
    word_indexes.append(encoded[i].word_ids())


for token, word_idx in zip(id_token[0], word_indexes[0]):
    print(f"Token: {token:<15} -> Word Index: {word_idx}")


