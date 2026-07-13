# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.exceptions import OutputParserException
# from pydantic import BaseModel, Field
# from typing import Literal
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

# class PersonInfo(BaseModel):
#     name: str = Field(description="Name of the employee")
#     job: str = Field(description="Job of the employee")
#     skill: str = Field(description="Skills the employee has")
#     salary: float = Field(description="Salary of the employee")
#     rating: float = Field(description="Rating of the employee", ge=1, le=5)
#     sentiment: Literal['positive', 'negative', 'neutral'] = Field(
#         description="Overall sentiment about the employee based on the text"
#     )

# class ExtractPersons(BaseModel):
#     person: list[PersonInfo] = Field(description="List of all individuals extracted from the text")


# parser = PydanticOutputParser(pydantic_object=ExtractPersons)

# examples = [
#     {
#         "input": "We looked at Alice Smith, a Python Developer. She makes 95000. Her Django skills are great. Everyone loves her attitude, giving her a solid 5 rating.",
#         "output": '{"person" : [{"name": "Alice Smith", "job": "Python Developer", "skill": "Python, Django", "salary": 95000.0, "rating": 5, "sentiment": "positive"}]}'
#     },
#     {
#         "input": "Bob Johnson is an Intern earning 40000. He knows basic HTML. His manager noted he is struggling to finish tasks on time, rating him 2, resulting in a neutral outlook.",
#         "output": '{"person" : [{"name": "Bob Johnson", "job": "Intern", "skill": "HTML", "salary": 40000.0, "rating": 2, "sentiment": "neutral"}]}'
#     }
# ]

# example_prompt = ChatPromptTemplate.from_messages([
#     ('human', '{input}'),
#     ('ai', '{output}')
# ])

# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=example_prompt,
#     examples=examples
# )

# final_prompt = ChatPromptTemplate.from_messages([
#     ('system', 'You are a helpful text extractor. Extract information into the given schema.'),
#     few_shot_prompt,
#     ('human', 'Here is the data:\n{data}')
# ]).partial(format_instructions=parser.get_format_instructions())

# chain = final_prompt | model| parser
# data = input("Enter user data: ")

# try:
#     response = chain.invoke({'data': data})

#     for p in range(len(response.person)):
#         print(f'{p+1}.', response.person[p].model_dump())
# except OutputParserException as e :
#     print(f'Error occured while parsing {e}')
# except Exception as e: 
#     print(f'An Error Occured {e}')




























from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, RootModel
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

class PersonInfo(BaseModel):
    name: str = Field(description="Name of the employee")
    job: str = Field(description="Job of the employee")
    skill: str = Field(description="Skills the employee has")
    salary: float = Field(description="Salary of the employee")
    rating: float = Field(description="Rating of the employee", ge=1, le=5)
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description="Overall sentiment about the employee based on the text")

class ExtractPersons(RootModel[list[PersonInfo]]):
    pass

parser = JsonOutputParser(pydantic_object=ExtractPersons)

examples = [
    {
        "input": "We looked at Alice Smith, a Python Developer. She makes 95000. Her Django skills are great. Everyone loves her attitude, giving her a solid 5 rating.",
        "output": '[{"name": "Alice Smith", "job": "Python Developer", "skill": "Python, Django", "salary": 95000.0, "rating": 5, "sentiment": "positive"}]'
    },
    {
        "input": "Bob Johnson is an Intern earning 40000. He knows basic HTML. His manager noted he is struggling to finish tasks on time, rating him 2, resulting in a neutral outlook.",
        "output": '[{"name": "Bob Johnson", "job": "Intern", "skill": "HTML", "salary": 40000.0, "rating": 2, "sentiment": "neutral"}]'
    }
]

example_prompt = ChatPromptTemplate.from_messages([
    ('human', '{input}'),
    ('ai', '{output}')
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful text extractor. Extract information into the given schema.\n{format_instructions}'),
    few_shot_prompt,
    ('human', 'Here is the data:\n{data}')
]).partial(format_instructions=parser.get_format_instructions())

try:
    chain = final_prompt | model | parser
    data = input("Enter user data: ")

    response = chain.invoke({'data': data})

    for p in range(len(response)):
        print(f'{p+1}.', response[p])
except OutputParserException as e:
    print(f"Error occured while parsing {e}")
except Exception as e: 
    print(f'An Error Occured {e}')



