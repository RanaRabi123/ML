from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import sys

sys.stdout.reconfigure(encoding='utf-8')


load_dotenv()
model = ChatGoogleGenerativeAI(
    model = 'gemini-3.1-flash-lite'
)

prompt_summarize = ChatPromptTemplate.from_messages([
    ('system', 'You are a text summarizer , You need to summarize the text given by user/human'),
    ('human', 'This is the text: \n{text}')
])

prompt_translate = ChatPromptTemplate.from_messages([
    ('system', 'You are a text translator. Translate the text into standard, native Urdu script. Do NOT put spaces between characters of a word. Ensure words are properly connected and readable in standard Urdu text flow.'),
    ('human', 'translate the given text: \n{text_trans}')
])


text =input('Enter the text : ')

chain_summ = prompt_summarize | model| StrOutputParser()

chain_trans = prompt_translate | model| StrOutputParser()


chain = {'text_trans':chain_summ} | chain_trans
try:
    response=chain.invoke({
        'text':text
        })

    with open("urdu_output.txt", "w", encoding="utf-8") as file:
        file.write(response)

    print(response)
except Exception as e:
    print(f"An error occurred: {e}")