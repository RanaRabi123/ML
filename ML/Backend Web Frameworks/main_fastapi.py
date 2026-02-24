from fastapi import FastAPI, Path, HTTPException, Query
from fastapi.responses import JSONResponse
import json
from pydantic import BaseModel, Field, computed_field 
from typing import Literal, Annotated, Optional

app = FastAPI()

class Patient(BaseModel):
    id : Annotated[str, Field(..., description='Enter the id of the patient', examples=['P001'])]
    name : Annotated[str, Field(..., description='Enter the name of the patient')]
    age: Annotated[int, Field(..., description='Enterthe age of the patient', gt=0 ,lt=120)]
    gender : Annotated[Literal['Male', 'Female', 'Others'], Field(..., description='Enter the Gender of the patient ')]
    city : Annotated[str, Field(..., description='Enter the city name where the patient lives ')]
    weight : Annotated[float, Field(...,gt=0,  description='Weight of the patient in KGs')] 
    height : Annotated[float, Field(...,gt=0, description='Height of the patient in Meters')]


    @computed_field
    @property
    def bmi(self) -> float:
        bmi = round((self.weight/self.height**2), 2)
        return bmi
    



class Patient_update(BaseModel):
    name : Annotated[Optional[str], Field(default=None)]
    age: Annotated[Optional[int], Field(default=None, gt=0, lt=120)]
    gender : Annotated[Optional[Literal['Male', 'Female', 'Others']], Field(default=None)]
    city : Annotated[Optional[str], Field(default=None)]
    weight : Annotated[Optional[float], Field(default=None)] 
    height : Annotated[Optional[float], Field(default=None)]



def load_data():
    with open('patients.json', 'r') as f:
        data = json.load(f)
        return data
    

def save_data(data):
    with open('patients.json', 'w') as f:
        json.dump(data, f)


@app.get('/')
def home():
    return {"Info": 'Patients Management System '}

@app.get('/about')
def about():
    return "Fully working API of the patients record "


@app.get('/view')
def view():
    data = load_data()
    return data


@app.get('/patient/{patient_id}')
def view_patient(patient_id: str = Path(..., description='Enter any patien id ', examples=['P001'])):
    data = load_data()
    if patient_id in data:
        return data[patient_id]
    
    raise HTTPException(status_code=404, detail='Patient item not found')



@app.get('/sort_by')
def sort_patient(sortby:str = Query(..., description='Enter on which feature u wnat to sort', examples=['age, weight, height']), 
                 orderby: str = Query('asc', description='Their sorted order will be asc OR des', examples=['asc, des'])):
    sort_values = ['age', 'weight', 'height']
    if sortby not in sort_values:
        raise HTTPException(status_code=400, detail=f'Wrong option, select from {sort_values}')
    if orderby not in ['asc', 'des']:
        raise HTTPException(status_code=400, detail="Wrong order, Select from asc OR des")
    
    reverse_order = True if orderby=='des' else False 
    data = load_data()
    sorted_data = sorted(data.values(), key=lambda x:x.get(sortby, 0), reverse=reverse_order)
    return sorted_data





@app.post('/create')
def create_patient(patient : Patient):
    data = load_data()

    if patient.id in data: 
        raise HTTPException(status_code=400, detail='Patient with this id already exists ')
    else:
        data[patient.id] = patient.model_dump(exclude=['id'])

              
            #  saving the file as JSON format 
    save_data(data)


    return JSONResponse(status_code=201, content={'message': 'Patient data created successfully ...'})






@app.put('/edit/{patient_id}')
def update_patient(patient_id: str, patient_update : Patient_update):
    data = load_data()

    if patient_id not in data:
        raise HTTPException(status_code=404, detail='This patient does not exist ...')
    else :
        existing_patient_info = data[patient_id]

            # here exclude means , if something is None by default it will ignore it 
        updated_patinet_info = patient_update.model_dump(exclude_unset=True)    

        for key, value in updated_patinet_info.items():
            existing_patient_info[key] = value


        # existing_patient_info -> pydantic object -> update bmi
        existing_patient_info['id'] = patient_id
        patient_pydantic_obj = Patient(**existing_patient_info)

        # pydantic object -> dictionary 
        existing_patient_info = patient_pydantic_obj.model_dump(exclude='id')

        # add this dictionary to data
        data[patient_id] = existing_patient_info

        save_data(data)


        return JSONResponse('Patient updated successfully ....')
    





@app.delete('/patient/{patient_id}')
def deleting_patient(patient_id: str):
    data = load_data()

    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    else:
        del data[patient_id]

    save_data(data)

    return JSONResponse('Patient Deleted successfully ....')