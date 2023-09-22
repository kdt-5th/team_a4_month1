import os
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import joblib
import pycountry_convert as pc

path = os.path.dirname(os.path.realpath(__file__))

def label_df(original_df):
    le_map = joblib.load(os.path.join(path,"model/labeling_map.joblib"))
    #print("라벨링맵")
    #print(type(le_map))

    le=LabelEncoder()
    obj_cols = original_df.select_dtypes(['object']).columns
    for col in obj_cols:
        if col in le_map:
            le.fit(le_map[col])
            try:
                original_df[col] = le.transform(original_df[col])
            except:
                print("에러: ",col)
                print(original_df[col])
                original_df[col]=0


    return original_df


def convert_to_pd(request_data):
    form_data=request_data.form
    input_features = ['experience_level','employment_type','company_size', 'jobs', 'remote_ratio']
    
    dicted_data={key: [form_data[key]] for key in input_features}

    try:
        #dicted_data['residence_continent'] = [coco.convert(names=form_data['addressCountry'], to='Continent')]

        #print("대륙?")
        #print(dicted_data['residence_continent'])
        dicted_data['residence_continent'] = pc.country_alpha2_to_continent_code(form_data['addressCountry'])
        dicted_data['residence_continent'] = [pc.convert_continent_code_to_continent_name(dicted_data['residence_continent'])]
    except:
        dicted_data['residence_continent'] = [""]

    dicted_data['Country_Grouped']= [form_data['addressCountry']] if form_data['addressCountry'] in ['US', 'GB', 'CA', 'ES', 'IN'] else ['Others']

    if dicted_data['residence_continent'][0]=='Oceania':
        dicted_data['residence_continent'][0]='Australia'

    df =pd.DataFrame(dicted_data)
    df = label_df(df)
    
    return df


def predict(form_data):
    features = ['experience_level','employment_type','jobs','residence_continent','remote_ratio','Country_Grouped','company_size']
    test=convert_to_pd(form_data)

    loaded_model = joblib.load(os.path.join(path,"model/model.pkl"))
    pred=loaded_model.predict(test[features])

    #print(f"예측:{pred}")

    return round(float(str(pred)[1:-1]))

