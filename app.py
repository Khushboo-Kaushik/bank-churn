import pandas as pd
import streamlit as st 
from tensorflow.keras.models import load_model
import pickle

model = load_model('bankchurnprediction.keras')

def pickler_reader(pickler_name ):
    with open(pickler_name ,'rb') as file:
        pickler = pickle.load(file)
    return pickler
cov_bool = {'No':0 ,'Yes':1}
st.title("Bank Keeper Churn Alert")
# CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
credit_score = st.slider("Select Credit score", min_value = 300, max_value = 900, step = 1)
geography = st.radio("Select Geography:", ('Spain', 'France', 'Germany'))
gender = st.radio("Select Gender:", ('Male', 'Female'))
age = st.slider("Select Age:",  min_value = 18, max_value = 120, step = 1)
tenure = st.slider("Select Tenure:", min_value = 0, max_value = 20, step = 1)
balance = st.slider("Select Balance", min_value = 0, max_value = 99999, step = 1)
no_of_products = st.slider("Select No. of products:", min_value = 0, max_value = 10, step = 1)
has_credit = st.radio("Has credit card", ('Yes', 'No'))
is_active = st.radio("Is member active", ('Yes', 'No'))
salary = st.slider("Enter Your salary",min_value = 0, max_value = 99999, step = 1)
analysis = st.button("Predict")
if analysis:
    user_input = {
        'CreditScore': credit_score,
        'Geography' : geography , 
        'Gender' : gender ,
        'Age' : age ,
        'Tenure' : tenure ,
        'Balance' : balance,
        'NumOfProducts' : no_of_products , 
        'HasCrCard' : cov_bool.get(has_credit) , 
        'IsActiveMember' : cov_bool.get(is_active),
        'EstimatedSalary' : salary
    }
    lable_encode = pickler_reader('Genderencoder.pkl')
    user_input['Gender'] = lable_encode.transform([user_input['Gender']])
    user_input = pd.DataFrame(user_input)
    one_hot_encode = pickler_reader('Geographyencoder.pkl')
    column_encode = one_hot_encode.transform(user_input[['Geography']]).toarray()
    column_encode_df = pd.DataFrame(column_encode , columns=one_hot_encode.get_feature_names_out(['Geography']))
    user_input = pd.concat([user_input.drop('Geography',axis=1 ) ,column_encode_df ],axis=1)
    scaller = pickler_reader('scallerencoder.pkl')
    scalled_data = scaller.transform(user_input)
    pridict = model.predict(scalled_data)  
    if pridict[0][0]*100 > 50:
        st.info(f'Customer Will Churn. Chances : {pridict[0][0] * 100:.2f} %'   )
    else:
        st.info(f"Customer will not churn. Chances: {pridict[0][0] * 100:.2f} %")