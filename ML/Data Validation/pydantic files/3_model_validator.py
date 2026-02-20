from pydantic import BaseModel, EmailStr, AnyUrl, model_validator
from typing import List, Dict, Optional
# model_validator --> it is same as field_validator, but in it we can validate muliple fields for each other 

class Patient_c(BaseModel):
    name: str
    age : int
    weight: float
    social_url : AnyUrl
    email : EmailStr
    allergies: Optional[List[str]]=None
    contact_info: Dict[str, str]



    @model_validator(mode='after')
    def validating_contact(cls, model):
        if model.age > 60 and 'emergency' not in model.contact_info:
            raise ValueError('pateint must has emergency info ...')
        else: 
            return model


def patient_view(obj_1: Patient_c):
    print(obj_1.name)
    print(obj_1.age)
    print(obj_1.weight)
    print(obj_1.email)
    print(obj_1.social_url)
    print(obj_1.allergies)
    print(obj_1.contact_info)

patient_values = {'name':'Rana', 'weight': 77.8, 'age':55, 'email':'rana@mzn.com', 'allergies':['sneezing', 'hunger'], 'social_url': 'https://youtube.com', 'contact_info':{'phone':'1234567890'}}


obj_1 = Patient_c(**patient_values)

patient_view(obj_1)