from pydantic import BaseModel, EmailStr, AnyUrl, computed_field
from typing import List, Dict, Optional
# computed_field --> it is used when two or more fields r used to compute the value of other, yet we can say that it is optional
class Patient_c(BaseModel):
    name: str
    age : int
    weight: float
    height: float
    social_url : AnyUrl
    email : EmailStr
    allergies: Optional[List[str]]=None
    contact_info: Dict[str, str]



    @computed_field
    @property
    def computing_bmi(self) -> float:      # arrow just tell the return type of the function
        bmi = round((self.weight/self.height**2), 2)
        return bmi


def patient_view(obj_1: Patient_c):
    print(obj_1.name)
    print(obj_1.age)
    print(obj_1.weight)
    print(obj_1.email)
    print(obj_1.social_url)
    print("BMI value ", obj_1.computing_bmi)
    print(obj_1.contact_info)

patient_values = {'name':'Rana', 'weight': 77.8, 'age':55, 'height': 5.5,  'email':'rana@mzn.com', 'allergies':['sneezing', 'hunger'], 'social_url': 'https://youtube.com', 'contact_info':{'phone':'1234567890'}}


obj_1 = Patient_c(**patient_values)

patient_view(obj_1)