from transformers import AutoTokenizer, AutoModelForSequenceClassification
from keybert import KeyBERT
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from docx import Document
from pypdf import PdfReader


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


def data_preprocessing(document):
    file_path = document
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            document = file.read()

    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        file = ''
        for para in doc.paragraphs:
            file+=para.text
        document = file
            

    elif file_path.endswith('.pdf'):
        file = PdfReader(file_path)
        file_content=''
        for index, page in enumerate(file.pages):
            file_content += page.extract_text()
        document = file_content

    else:
        raise ValueError("Unsupported document type. Please provide a string, docx.Document, or pdfplumber.PDF.")

    return document



def document_classification(user_file):

    document = data_preprocessing(user_file)

    # Tokenize the input document
    inputs = tokenizer(document, return_tensors="pt", truncation=True, padding=True)
    
    # Get the model's predictions
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

        return print("Extracted Summarization is :", keywords[0][0])
    

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

        return print(tokenizer.decode(output[0], skip_special_tokens=True))
    


user = input('Enter any text : ' )
summarize_document(user)