                #  stroing chat as it is 

# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import HumanMessage , AIMessage
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatGroq(
#     model='llama-3.3-70b-versatile'
# )

# memory = []
# max_length_memory= 10
# prompt = ChatPromptTemplate([
#     ('system', 'you are a helpful assistant,use the conversation history to answer the user question'),
#     MessagesPlaceholder(variable_name='memory')
# ])

# chain = prompt | model

# while True:
#     task = input('User : What you want  to do : ')
    
#     if task.lower().strip() =='exit':
#         break
#     else:
#         memory.append(HumanMessage(content=task))
#         if len(memory) > max_length_memory:
#             memory = memory[-max_length_memory:]

#         response = chain.invoke(
#             {'memory': memory}
#         )
#         memory.append(AIMessage(content=response.content))

#         print('AI : ', response.content)
































                #  summarizing earlier memory 

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage , AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model='llama-3.3-70b-versatile'
)

memory = []
max_length_memory= 10
prompt = ChatPromptTemplate([
    ('system', 'you are a helpful assistant,use the conversation history to answer the user question'),
    MessagesPlaceholder(variable_name='memory')
])

chain = prompt | model

def memory_summarized(messages):
    summarized_prompt = ChatPromptTemplate.from_messages([('system', 'You need to summarize the given input, make sure that key facts , names and important things should keep'),
                                            MessagesPlaceholder(variable_name='messages')])
    return (summarized_prompt| model).invoke({'messages':messages}).content


while True:
    task = input('User : What you want  to do : ')
    
    if task.lower().strip() =='exit':
        break
    else:
        memory.append(HumanMessage(content=task))
        if len(memory) > max_length_memory:
            to_keep = memory[-2:]
            to_summarize = memory[:-2]
            summary_text = memory_summarized(to_summarize)
            memory = [SystemMessage(content=f'Summary of earlier converstion {summary_text}')] + to_keep

        response = chain.invoke(
            {'memory': memory}
        )
        memory.append(AIMessage(content=response.content))

        print('AI : ', response.content)

print(memory)