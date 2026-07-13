# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate , PromptTemplate
# load_dotenv()

# llm = ChatGroq(
#     model = 'llama-3.3-70b-versatile'
# )

# prompt = PromptTemplate(
#     template = """You are a helpful assistant, you need to generate on {topic}, with {tone} tone and it's length should be {length}""",
#     input_variables=['topic', 'tone', 'length']
# )

# format_msg = prompt.format(
#     topic='briefly tell, what is langchain',
#                         tone='easy',
#                         length= 'short'

# )

# result = llm.invoke(format_msg)
# print(result.content)





















# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate , PromptTemplate
# load_dotenv()

# llm = ChatGroq(
#     model = 'llama-3.3-70b-versatile'
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "you are a helpful assistant"),
#     ('user', 'generate on{topic} , with {tone} tone and of {length} length')],
#     )

# format_msg = prompt.format(
#                         topic='briefly tell, what is langchain',
#                         tone='easy',
#                         length= 'short'

# )

# result = llm.invoke(format_msg)
# print(result.content)


































# from langchain_ollama import ChatOllama
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate 
# load_dotenv()

# llm = ChatOllama(
#     model="llama3.1:8b",
#     keep_alive='60m'
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "you are a helpful assistant"),
#     ('user', 'generate on{topic} , with {tone} tone and of {length} length')],
#     )

# chain = prompt | llm


# topic = input("what you want to do : ")
# tone = input("what can be the tone of the response: ")
# length = input("what can be it's length  : ")
# result = chain.invoke(
#                     {'topic':topic,
#                         'tone':tone,
#                         'length':length})

# print(result.content)






























from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate , FewShotChatMessagePromptTemplate
load_dotenv()

llm = ChatOllama(
    model="llama3.1:8b",
    keep_alive='60m'
)

examples = [
    {
        "topic": "a productivity app",
        "tone": "playful",
        "length": "short",
        "output": "Meet TaskZap — the to-do list that actually gets things done, so you don't have to nag yourself twice."
    },
    {
        "topic": "a cybersecurity course",
        "tone": "formal",
        "length": "short",
        "output": "This course equips professionals with the practical skills required to identify, assess, and mitigate modern cybersecurity threats."
    },
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Topic: {topic}\nTone: {tone}\nLength: {length}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a skilled copywriter. Study the examples, then generate new content matching the requested topic, tone, and length."),
    few_shot_prompt,
    ("human", "Topic: {topic}\nTone: {tone}\nLength: {length}"),
])


chain = final_prompt | llm

topic = input("what you want to do : ")
tone = input("what can be the tone of the response: ")
length = input("what can be it's length  : ")

result = chain.invoke(
                    {'topic':topic,
                        'tone':tone,
                        'length':length})

print(result.content)