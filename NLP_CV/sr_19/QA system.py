import kagglehub

# Download latest version
path = kagglehub.dataset_download("andreajaunarena/triviaqa-dataset")

from transformer import TFAutoModelForQuestionAnswering, TFAutoTokenizer


tokenizer = TFAutoTokenizer('bert-large-uncased-whole-word-masking-finetuned-squad')
model = TFAutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

