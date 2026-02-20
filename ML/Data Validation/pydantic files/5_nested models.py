from pydantic import BaseModel

class address(BaseModel):
    city : str
    state : str
    zipcode : str

class patient(BaseModel):
    name : str
    age : int
    address : address

value_address = {'city':'lahore', 'state':'punjab', 'zipcode':'54500'}
obj_address = address(**value_address)

value_patient = {'name':'Rana', 'age': 22, 'address':obj_address}
obj_patient = patient(**value_patient)

print(obj_patient.name)
print(obj_patient.age)
print(obj_patient.address.city)
print(obj_patient.address.zipcode)
print(obj_patient.address.state)