from pydantic import BaseModel

class address(BaseModel):
    city : str
    state : str
    zipcode : str

class patient(BaseModel):
    name : str
    age : int
    gender : str
    address : address

value_address = {'city':'lahore', 'state':'punjab', 'zipcode':'54500'}
obj_address = address(**value_address)

value_patient = {'name':'Rana', 'gender':'male', 'age': 22, 'address':obj_address}
obj_patient = patient(**value_patient)

# print(obj_patient.name)
# print(obj_patient.age)
# print(obj_patient.address.city)
# print(obj_patient.address.zipcode)
# print(obj_patient.address.state)





            # exporting pydantic model 
            # as a dictionary 
temp_dict = obj_patient.model_dump(include={'address':['state', 'city']})
print(temp_dict)
print(type(temp_dict))



            #  as a  JSON format
temp_json = obj_patient.model_dump_json(include=['name', 'gender'])
print(temp_json)
print(type(temp_json))      # string