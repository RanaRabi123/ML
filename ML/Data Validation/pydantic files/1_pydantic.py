from pydantic import BaseModel, EmailStr, AnyUrl, Field
from typing import List, Dict, Optional, Annotated
# field --> only used when basemodel class is inherited 
# annotated --> can be used where basemodel is not inherited , and used for data validation , it is modern way style ,
# annotated, --> such as query can perfom the same but using it via anotated is good practice


class Patient_c(BaseModel):
    name: str
    age : Annotated[int, Field(description="it should not be greate than 120", lt=120)]
    weight: float
    social_url : AnyUrl
    email : EmailStr
    allergies: Optional[List[str]]=Field(default=None, max_length=4, min_length=2)
    contact_info: Dict[str, str]
    

def patient_view(obj_1: Patient_c):
    print(obj_1.name)
    print(obj_1.age)
    print(obj_1.weight)
    print(obj_1.email)
    print(obj_1.social_url)
    print(obj_1.allergies)
    print(obj_1.contact_info)

patient_values = {'name':'Rana', 'weight': 77.8, 'age':119, 'email':'rana@gmail.com', 'allergies':['sneezing', 'hunger'], 'social_url': 'https://youtube.com', 'contact_info':{'phone':'1234567890'}}
obj_1 = Patient_c(**patient_values)

patient_view(obj_1)