from flask import Flask, jsonify, render_template, request
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import country_converter as coco
import warnings
import os, sys
import copy

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from predict_salary import predict


plt.style.use('fivethirtyeight')

app = Flask(__name__)

#현재 파일의 절대경로
path = os.path.dirname(os.path.abspath(__file__)) 


@app.route('/')
def index():
    static_path = 'static/'

    image_paths = []

    graphs = []
    # 데이터 불러오기
    
    # original_df = pd.read_csv('./archive/ds_salaries.csv')
    #original_df = pd.read_csv('C:/Users/Jena_laptop/Desktop/programers_kdt/project/data_science_salaries/data_science_salaries/archive/ds_salaries.csv')
    original_df = pd.read_csv(os.path.join(path, "archive/ds_salaries.csv"))


    # 작업 분류
    # 직업과 해당 범주를 딕셔너리로 매핑
    job_category_mapping = {
        'Data Scientist': '데이터 사이언티스트',
        'Applied Scientist': '데이터 사이언티스트',
        'Research Scientist': '데이터 사이언티스트',
        'Applied Data Scientist': '데이터 사이언티스트',
        'Staff Data Scientist': '데이터 사이언티스트',
        'Product Data Scientist': '데이터 사이언티스트',
        'Principal Data Scientist' : '데이터 사이언티스트',
        'Lead Data Scientist' : '데이터 사이언티스트',
        'Data Science Lead' : '데이터 사이언티스트',
        'Data Science Consultant' : '데이터 사이언티스트',
        'Data Science Engineer' : '데이터 사이언티스트',
        'Data Science Tech Lead' : '데이터 사이언티스트',
        'Data Scientist Lead' : '데이터 사이언티스트',

        'ML Engineer': '머신 러닝 엔지니어',
        'Machine Learning Engineer': '머신 러닝 엔지니어',
        'Applied Machine Learning Engineer': '머신 러닝 엔지니어',
        'Machine Learning Research Engineer': '머신 러닝 엔지니어',
        'Principal Machine Learning Engineer': '머신 러닝 엔지니어',
        'Lead Machine Learning Engineer': '머신 러닝 엔지니어',
        'Machine Learning Manager': '머신 러닝 엔지니어',
        'Machine Learning Developer': '머신 러닝 엔지니어',
        'Machine Learning Infrastructure Engineer': '머신 러닝 엔지니어',
        'Machine Learning Researcher': '머신 러닝 엔지니어',
        'MLOps Engineer': '머신 러닝 엔지니어',
        'Research Engineer' : '머신 러닝 엔지니어',
        'Machine Learning Scientist' : '머신 러닝 엔지니어',
        'Applied Machine Learning Scientist' : '머신 러닝 엔지니어',
        'Machine Learning Software Engineer' : '머신 러닝 엔지니어',


        'Marketing Data Engineer': '데이터 엔지니어',
        'Data Engineer': '데이터 엔지니어',
        'ETL Engineer': '데이터 엔지니어',
        'Data DevOps Engineer': '데이터 엔지니어',
        'Cloud Database Engineer': '데이터 엔지니어',
        'Azure Data Engineer': '데이터 엔지니어',
        'Cloud Data Engineer': '데이터 엔지니어',
        'ETL Developer': '데이터 엔지니어',
        'Cloud Data Architect': '데이터 엔지니어',
        'Lead Data Engineer': '데이터 엔지니어',
        'Principal Data Engineer': '데이터 엔지니어',
        'Analytics Engineer' : '데이터 엔지니어',
        'Director of Data Science' : '데이터 엔지니어',
        'Data Specialist' : '데이터 엔지니어',
        'Data Infrastructure Engineer' : '데이터 엔지니어',
        'Software Data Engineer' : '데이터 엔지니어',

        'Data Analyst': '비즈니스 인텔리전스 및 분석',
        'Data Analytics Manager': '비즈니스 인텔리전스 및 분석',
        'Business Data Analyst': '비즈니스 인텔리전스 및 분석',
        'BI Data Engineer': '비즈니스 인텔리전스 및 분석',
        'BI Developer': '비즈니스 인텔리전스 및 분석',
        'BI Data Analyst': '비즈니스 인텔리전스 및 분석',
        'BI Analyst': '비즈니스 인텔리전스 및 분석',
        'Data Analytics Specialist': '비즈니스 인텔리전스 및 분석',
        'Data Analytics Engineer': '비즈니스 인텔리전스 및 분석',
        'Data Analytics Consultant': '비즈니스 인텔리전스 및 분석',
        'Power BI Developer': '비즈니스 인텔리전스 및 분석',
        'Business Intelligence Engineer' : '비즈니스 인텔리전스 및 분석',
        
        'NLP Engineer': '자연어 처리 및 음성 처리',
        
        'Computer Vision Engineer': '컴퓨터 비전(CV) 종사자',
        'Computer Vision Software Engineer': '컴퓨터 비전(CV) 종사자',
        '3D Computer Vision Researcher': '컴퓨터 비전(CV) 종사자',

        'Deep Learning Researcher': '딥 러닝 엔지니어',
        'Deep Learning Engineer': '딥 러닝 엔지니어',

        'Big Data Engineer': '빅 데이터 분석가',
        'Big Data Architect': '빅 데이터 분석가',

        'Data Architect': '데이터 아키텍터',
        'Principal Data Architect': '데이터 아키텍터',

        'Data Quality Analyst': '데이터 관리 및 운영',
        'Compliance Data Analyst': '데이터 관리 및 운영',
        'Data Operations Engineer': '데이터 관리 및 운영',
        'Data Management Specialist': '데이터 관리 및 운영',
        'Data Operations Analyst': '데이터 관리 및 운영',
        'Manager Data Management' : '데이터 관리 및 운영',

        'Insight Analyst': '인사이트 분석',
        
        'Marketing Data Analyst': '데이터 분석가',
        'Product Data Analyst': '데이터 분석가',
        'Financial Data Analyst': '데이터 분석가',
        'Finance Data Analyst': '데이터 분석가',
        'Data Modeler' : '데이터 분석가',
        'Staff Data Analyst' : '데이터 분석가',
        'Lead Data Analyst' : '데이터 분석가',
        'Data Analytics Lead' : '데이터 분석가',
    
        'AI Developer': '인공 지능 개발자',
        'AI Scientist': '인공 지능 개발자',
        'AI Programmer' : '인공 지능 개발자',

        'Autonomous Vehicle Technician': '자율 주행 자동차 엔지니어',

        'Data Strategist': '데이터 전략가 및 리더',
        'Head of Data': '데이터 전략가 및 리더',
        'Data Science Manager': '데이터 전략가 및 리더',
        'Data Manager': '데이터 전략가 및 리더',
        'Head of Data Science': '데이터 전략가 및 리더',
        'Data Lead': '데이터 전략가 및 리더',
        'Head of Machine Learning': '데이터 전략가 및 리더',
        'Principal Data Analyst': '데이터 전략가 및 리더',
        'Staff Data Scientist': '데이터 전략가 및 리더'
    }

    original_df['jobs'] = original_df['job_title'].apply(lambda x : job_category_mapping[x])
    original_df['jobs']

    original_df = original_df.drop(['salary','salary_currency'], axis=1)

    original_df['residence_continent']=original_df['employee_residence']

    original_df['residence_continent']=original_df['residence_continent'].replace(( 'AS', 'CA', 'CR', 'DO', 'SV', 'GT', 'HN', 'MX', 'NI', 'PA', 'PR', 'US'), "North America")
    original_df['residence_continent']=original_df['residence_continent'].replace(('NG', 'GH', 'KE', 'ZA', 'DZ', 'EG', 'MA','TN','CF'), "Africa")
    original_df['residence_continent']=original_df['residence_continent'].replace(('CN', 'HK', 'IN', 'IL', 'IR', 'IQ', 'JP', 'KW', 'KZ', 'MY', 'PH', 'PK', 'SA', 'SG', 'KR', 'AE', 'TH', 'TR', 'UZ', 'VN','AM','ID', ), "Asia")
    original_df['residence_continent']=original_df['residence_continent'].replace(('AU', 'NZ','AS'), "Australia")
    original_df['residence_continent']=original_df['residence_continent'].replace(('AL', 'AT', 'BE', 'BA', 'BG', 'HR', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IS', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'GB', 'UA', 'RU', 'CY','MK','JE','MD'), "Europe")
    original_df['residence_continent']=original_df['residence_continent'].replace(('AR', 'BO', 'BR', 'CL', 'CO', 'EC', 'PY', 'PE', 'UY', 'VE'), "South America")

    original_df['Country_Grouped'] = original_df['employee_residence'].apply(lambda x: x if x in ['US', 'GB', 'CA', 'ES', 'IN'] else 'Others')
    original_df['Company_Grouped'] = original_df['company_location'].apply(lambda x: x if x in ['US', 'GB', 'CA', 'ES', 'IN'] else 'Others')
    # original_df['Salary_Grouped'] = original_df['salary_currency'].apply(lambda x: x if x in ['USD', 'EUR', 'GBP', 'INR', 'CAD'] else 'Others')

    # 가독성을 위한 라벨링
    original_df['experience_level'] = original_df['experience_level'].replace('EN', 'Junior')
    original_df['experience_level'] = original_df['experience_level'].replace('MI', 'Senior')
    original_df['experience_level'] = original_df['experience_level'].replace('SE', 'Expert')
    original_df['experience_level'] = original_df['experience_level'].replace('EX', 'Director')

    original_df['employment_type'] = original_df['employment_type'].replace('FT', 'Full-Time')
    original_df['employment_type'] = original_df['employment_type'].replace('CT', 'Contract')
    original_df['employment_type'] = original_df['employment_type'].replace('FL', 'Freelance')
    original_df['employment_type'] = original_df['employment_type'].replace('PT', 'Part-Time')


    #---------------------------------------------------------
    # 고용 인구 거주지
    original_df['employee_residence'] = coco.convert(original_df['employee_residence'], to='ISO3')

    residence = original_df['employee_residence'].value_counts()
    fig = px.choropleth(locations=residence.index,
                    color=residence.values,
                    color_continuous_scale=px.colors.sequential.Agsunset,
                    template='simple_white',
                    title='Employee Residence Distribution Map',)

    fig.update_layout(title_x = 0.5)

    # 그래프를 HTML 형태로 변환
    graphs.append(pio.to_html(fig))
    #---------------------------------------------------------
    
    fig, ax = plt.subplots(3,3, figsize=(15,15))
    for row in ax:
        for subplot in row:
            subplot.tick_params(axis='x', labelrotation=45)
    sns.countplot(original_df, x='work_year', ax=ax[0,0])
    sns.countplot(original_df, x='experience_level', ax=ax[0,1])
    sns.countplot(original_df, x='employment_type', ax=ax[0,2])
    sns.histplot(original_df, x='salary_in_usd', kde=True, ax=ax[1,0])
    sns.countplot(original_df, x='Country_Grouped', ax=ax[1,1])
    sns.countplot(original_df, x='remote_ratio', ax=ax[1,2])
    sns.countplot(original_df, x='Company_Grouped', ax=ax[2,0])
    sns.countplot(original_df, x='company_size', ax=ax[2,1])
    sns.countplot(original_df, x='residence_continent', ax=ax[2,2])

    plt.xticks(rotation=45)
    plt.tight_layout()

    image_path = static_path + 'seaborn_plot_1.png'
    plt.savefig(image_path)
    plt.close()
    image_paths.append(image_path)

    #---------------------------------------------------------
    level_avg = original_df[['salary_in_usd', 'experience_level']].groupby('experience_level').mean()

    fig = px.pie(level_avg, values=level_avg['salary_in_usd'], names=level_avg.index, height=600)

    fig.update_traces(hole=.3, textinfo='percent+label')
    graphs.append(pio.to_html(fig))

    #---------------------------------------------------------
    # - 'work_year'별 'experience_level'기준 salary 변화
    
    fig = px.line(original_df, x='work_year', y='salary_in_usd', color='experience_level', line_shape='linear')

    # 제목, 레이블, 범례 폰트 크기 등의 추가 설정
    fig.update_layout(
        title="Salary vs. Work Year based on Experience Level",
        xaxis_title="Work Year",
        yaxis_title="Salary in USD",
        legend_title="Experience Level",
        legend_font_size=12
    )

    # HTML로 변환
    image_path = static_path + 'seaborn_plot_2.png'
    plt.savefig(image_path)
    plt.close()
    image_paths.append(image_path)
    #---------------------------------------------------------
    # Job 분포
    total = len(original_df)
    job_per = original_df['jobs'].value_counts().apply(lambda x : round(x/total * 100,2))
    job_per=job_per[:10]

    fig = px.bar(y= job_per.values,
                x = job_per.index,
                color= job_per.index,
                color_discrete_sequence=px.colors.qualitative.G10,
                text = job_per.values,
                title='Top10 Jobs',
                template='ggplot2'
                )

    fig.update_layout(
        title_text='Job Distribution(%)',  
        height=650,  
        xaxis_title="Job Titles", 
        yaxis_title="Count",  
        font=dict(size=17, family="Franklin Gothic") 
    )

    graphs.append(pio.to_html(fig))

    #---------------------------------------------------------
    total = len(original_df)
    job_per = original_df['jobs'].value_counts().apply(lambda x : round(x/total * 100,2))
    job_per=job_per.astype(str)+'%'


    avg_sal = original_df.groupby('jobs')['salary_in_usd'].mean().apply(lambda x : round(x,2))
    avg_sal= avg_sal.sort_values( ascending=False)[:10]

    sal_per = pd.DataFrame(avg_sal).join(job_per)

    fig = px.bar(y= sal_per['salary_in_usd'],
                x = sal_per.index,
                color= sal_per.index,
                color_discrete_sequence=px.colors.qualitative.G10,
                text = sal_per['count'],
                title='Top10 Jobs - Salary',
                template='ggplot2'
                )

    fig.update_layout(
        title_text='Job Ratio-Salary',
        height=650,     
        xaxis_title="Job Titles",  
        yaxis_title="Salary",  
        font=dict(size=17, family="Franklin Gothic")  
    )

    graphs.append(pio.to_html(fig))

    #---------------------------------------------------------
    # 전세계 기준
    fig, ax = plt.subplots(3,3, figsize=(15,12))

    for row in ax:
        for subplot in row:
            subplot.tick_params(axis='x', labelrotation=45)
            
    sns.barplot(original_df, y='salary_in_usd',x='work_year', ax=ax[0,0])
    sns.barplot(original_df, y='salary_in_usd',x='experience_level', ax=ax[0,1])
    sns.barplot(original_df, y='salary_in_usd',x='employment_type', ax=ax[0,2])
    sns.histplot(original_df, x='salary_in_usd', kde=True, ax=ax[1,0])

    sns.barplot(original_df, y='salary_in_usd',x='Country_Grouped', ax=ax[1,1])
    sns.barplot(original_df, y='salary_in_usd',x='remote_ratio', ax=ax[1,2])
    sns.barplot(original_df, y='salary_in_usd',x='Company_Grouped', ax=ax[2,0])
    sns.barplot(original_df, y='salary_in_usd',x='company_size', ax=ax[2,1])
    sns.barplot(original_df, y='salary_in_usd',x='residence_continent', ax=ax[2,2])

    for i in range(3):
        for j in range(3):
            if (i, j) != (1, 0):
                for p in ax[i, j].patches:
                    ax[i, j].annotate(format(p.get_height(), '.2f'), 
                                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                                    ha='center', va='center', 
                                    xytext=(0, 10), 
                                    textcoords='offset points')

    plt.tight_layout()
    
    image_path = static_path + 'seaborn_plot_3.png'
    plt.savefig(image_path)
    plt.close()
    image_paths.append(image_path)

    #---------------------------------------------------------
    original_df['company_location'] = coco.convert(original_df['employee_residence'], to='ISO3')
    company_salary = original_df.groupby('company_location')['salary_in_usd'].mean()
    
    fig = px.choropleth(locations=company_salary.index,
                    color=company_salary.values,
                    hover_name= company_salary.index,
        
                    color_continuous_scale=px.colors.sequential.YlGnBu,
                    title='Company Location - Mean Salary Distribution Map')

    graphs.append(pio.to_html(fig))

    #---------------------------------------------------------
    exp_idx = original_df.experience_level.unique()

    plt.figure(figsize=(10,6))
    sns.set_style('darkgrid')
    sns.kdeplot(original_df['salary_in_usd'][original_df.experience_level == 'Junior'] , bw=.5)
    sns.kdeplot(original_df['salary_in_usd'][original_df.experience_level == 'Senior'], bw=.5)
    sns.kdeplot(original_df['salary_in_usd'][original_df.experience_level == 'Expert'], bw=.5)
    sns.kdeplot(original_df['salary_in_usd'][original_df.experience_level == 'Director'], bw=.5)

    plt.legend(['Junior','Senior','Expert','Director'])
    image_path = static_path + 'seaborn_plot_4.png'
    plt.savefig(image_path)
    plt.close()
    image_paths.append(image_path)
    #---------------------------------------------------------
    fig = px.box(original_df, x='experience_level', y='salary_in_usd', color='company_size', height=600, width=1100)
    graphs.append(pio.to_html(fig))
    #---------------------------------------------------------
    from sklearn.preprocessing import LabelEncoder 

    labeled_df = original_df
    obj_cols = labeled_df.select_dtypes(['object']).columns

    le = LabelEncoder()
    for col in obj_cols:
        labeled_df[col] = le.fit_transform(labeled_df[col])

    plt.figure(figsize=(10, 10))
    sns.heatmap(labeled_df.corr(), annot=True,fmt = '.2f', linewidths=.5, cmap='Blues')
    
    image_path = static_path + 'seaborn_plot_5.png'
    plt.savefig(image_path)
    plt.close()
    image_paths.append(image_path)
    #---------------------------------------------------------
    plt.figure(figsize=(10, 10))
    sns.heatmap(labeled_df.corr(), annot=True,fmt = '.2f', linewidths=.5, cmap='Blues')
    image_path = static_path + 'seaborn_plot_6.png'
    plt.savefig(image_path)
    plt.close()
    image_paths.append(image_path)

    
    return render_template('index.html', graph_html_list=graphs, image_paths=image_paths)

# 예측값 계산용
@app.route('/submit/', methods=['POST'])
def predict_calc():
    salary = predict(request)
    return jsonify({"val":salary})


@app.route('/predict')
def predict_sal():
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
