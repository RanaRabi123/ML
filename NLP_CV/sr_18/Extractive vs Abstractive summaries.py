import kagglehub
import pandas as pd
import numpy as np 
import os 
import regex as re 

path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")
file_path = os.path.join(path, os.listdir(path)[0], 'train.csv')

df = pd.read_csv(file_path)
df = df.sample(70000, random_state=33).reset_index(drop=True)

df.drop('id', axis=1, inplace=True)


article = []
highlight = []
for i in range(len(df)):
  article.append(re.sub("@\S+|--|CNN|^.*?UPDATED:\s*\.\s*.*?\d{4}\s*\.\s*|\n|\'", '', df['article'][i]))
  highlight.append(re.sub('\n', '', df['highlights'][i]))






                #  ti si used ot extract important  keywords 
from keybert import KeyBERT

# Initialize KeyBERT (uses a default sentence-transformer model)
kw_model = KeyBERT()
sentence = input('Enter any text')

# Extract the top 3 keywords/keyphrases, range measn how many words should need to extract
keywords = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1,3), top_n=3)

print("Extracted Keywords:", keywords)










from transformers import PegasusTokenizer, PegasusForConditionalGeneration

tokenizer = PegasusTokenizer.from_pretrained("human-centered-summarization/financial-summarization-pegasus")
model = PegasusForConditionalGeneration.from_pretrained("human-centered-summarization/financial-summarization-pegasus")


text_to_summarize = input('Enter any text : ')

input_ids = tokenizer(text_to_summarize, return_tensors="pt").input_ids
output = model.generate(
    input_ids, 
    max_length=32, 
    num_beams=5, 
    early_stopping=True
)

print(tokenizer.decode(output[0], skip_special_tokens=True))









from rouge_score import rouge_scorer

reference_summary ='Real sentence'
candidate_summary = 'generated summary '

# 2. Initialize the scorer with the specific ROUGE metrics you want to evaluate
# Using use_stemmer=True helps match different forms of the same word root
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# 3. Compute the scores
scores = scorer.score(reference_summary, candidate_summary)

# 4. Print the raw score named tuples (Precision, Recall, F-measure)
print("--- Detailed Scores ---")
for metric, score in scores.items():
    print(f"{metric.upper()}:")
    print(f"  Precision: {score.precision:.4f}")
    print(f"  Recall:    {score.recall:.4f}")
    print(f"  F1-Score:  {score.fmeasure:.4f}")
