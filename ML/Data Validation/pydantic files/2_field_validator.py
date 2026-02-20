from pydantic import BaseModel, EmailStr, AnyUrl, field_validator
from typing import List, Dict, Optional
# field_validator --> it is used to madify/transform value, or to raise custom errors , that file cannot do 

class Patient_c(BaseModel):
    name: str
    age : int
    weight: float
    social_url : AnyUrl
    email : EmailStr
    allergies: Optional[List[str]]=None
    contact_info: Dict[str, str]


    @field_validator('email')
    @classmethod
    def email_validate(cls, value):
        validation = ['bop.com', 'mzn.com', 'ue.com']
        domain_name = value.split('@')[-1]
        if domain_name not in validation:
            raise ValueError(f'Wrong domain name , enter only valid domains {validation}')
        else:
            return value



    @field_validator('name')
    @classmethod
    def name_validate(cls, value):
        return value.upper()



    @field_validator('age')
    @classmethod
    def age_validate(cls, value):
        if 0 < value < 100 : 
            return value
        else: 
            raise ValueError('Invalid Age, Age should be between 0 to 100')
        

def patient_view(obj_1: Patient_c):
    print(obj_1.name)
    print(obj_1.age)
    print(obj_1.weight)
    print(obj_1.email)
    print(obj_1.social_url)
    print(obj_1.allergies)
    print(obj_1.contact_info)

patient_values = {'name':'Rana', 'weight': 77.8, 'age':99, 'email':'rana@mzn.com', 'allergies':['sneezing', 'hunger'], 'social_url': 'https://youtube.com', 'contact_info':{'phone':'1234567890'}}

obj_1 = Patient_c(**patient_values)

patient_view(obj_1)