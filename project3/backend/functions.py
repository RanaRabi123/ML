from transformers import AutoTokenizer, AutoModelForSequenceClassification
from keybert import KeyBERT
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

label2id = {
    "legal": 0,
    "medical": 1,
    "news": 2,
    "entertainment": 3,
    "sports": 4,
    "technology": 5,
    "politics": 6,
    "education": 7
}

id2label = {value: key for key, value in label2id.items()}

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

def document_classification(user_file):

    inputs = tokenizer(user_file, return_tensors="pt", truncation=True, padding=True)
    
    outputs = model(**inputs)
    
    # Get the predicted class
    predicted_class = outputs.logits.argmax().item()
    predicted_label = id2label.get(predicted_class, "Unknown")
    return predicted_class, predicted_label

def summarize_document(user_file):
    _, predicted_label = document_classification(user_file)
    if predicted_label in ["medical", "legal", "health"]:  # only for these we are using extractive summarization

        print('extractive')
        kw_model = KeyBERT()

        # Extract the top 3 keywords/keyphrases, range measn how many words should need to extract
        keywords = kw_model.extract_keywords(user_file, keyphrase_ngram_range=(1,32), top_n=1)

        return keywords[0][0]
    

    else:
        print('abstractive')

        tokenizer = PegasusTokenizer.from_pretrained("human-centered-summarization/financial-summarization-pegasus")
        model = PegasusForConditionalGeneration.from_pretrained("human-centered-summarization/financial-summarization-pegasus")

        input_ids = tokenizer(user_file, return_tensors="pt", truncation=True).input_ids
        output = model.generate(
            input_ids,
            max_length=64,
            num_beams=5,
            early_stopping=True
        )

        return tokenizer.decode(output[0], skip_special_tokens=True)
    