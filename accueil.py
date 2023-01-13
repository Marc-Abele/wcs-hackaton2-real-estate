import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


df = pd.read_csv('df_final_hackaton.csv')

X = df[(df['type_local'] == 'Appartement') & (df['code_departement'] == 75)][['surface_reelle_bati', 'nombre_pieces_principales']]
y = df[(df['type_local'] == 'Appartement') & (df['code_departement'] == 75)]['valeur_fonciere']

outliers = ['surface_reelle_bati', 'nombre_pieces_principales']

arrond = df[df['code_departement'] == 75]['nom_commune'].unique()

ct = ColumnTransformer(transformers = [('rs', RobustScaler(), outliers)],
                       remainder = 'passthrough')

X_scaled = ct.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size = 0.8, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mae = round(mean_absolute_error(y_test, y_pred))

st.header('Bienvenue dans votre estimateur immobilier spécialisé dans les appartements parisiens')
st.title('')
st.subheader('Entrez les caractéristiques de votre bien')


arrondissement = st.selectbox('Localisation :', options = arrond)
st.write('')
surface = st.slider('Surface du bien (m²) :', min_value = int(X['surface_reelle_bati'].min()), max_value = 500)
st.write('')

col1, col2 = st.columns(2)

with col1:
    pieces = st.number_input('Nombre de pièces :', min_value = 1)
    st.write('')
    st.write('')
    garage = st.radio('Garage', ('Oui', 'Non'))
    st.write('')
    st.write('')   
    ascenseur = st.radio('Ascenseur', ('Oui', 'Non'))
    
    
with col2:
    chambres = st.number_input('Nombre de chambres :', min_value = 1, max_value = pieces)
    st.write('')
    st.write('')
    ext = st.radio('Balcon / Terrasse', ('Oui', 'Non'))
    st.write('')
    st.write('')   
    cave = st.radio('Cave', ('Oui', 'Non'))

estimation = int(lr.predict([[surface, pieces]]))

def formate(nb):
    return str("{:,}".format(nb))

cher = ['Paris 8e Arrondissement','Paris 7e Arrondissement','Paris 16e Arrondissement','Paris 6e Arrondissement','Paris 1er Arrondissement','Paris 4e Arrondissement']
moyen = ['Paris 3e Arrondissement','Paris 2e Arrondissement','Paris 9e Arrondissement','Paris 17e Arrondissement','Paris 5e Arrondissement','Paris 15e Arrondissement']



st.title('')
if st.button('Estimez mon bien'):
    if arrondissement in cher :
        st.subheader(('Votre bien immobilier est estimé entre ' + str(formate(1.2*math.trunc((estimation - 25000)))) + '€  et ' + str(formate(1.2*math.trunc((estimation + 25000)))) + '€.'))
    elif arrondissement in moyen :
        st.subheader(('Votre bien immobilier est estimé entre ' + str(formate(1.1*math.trunc((estimation - 25000)))) + '€  et ' + str(formate(1.1*math.trunc((estimation + 25000)))) + '€.'))
    else :
        st.subheader(('Votre bien immobilier est estimé entre ' + str(formate(math.trunc((estimation - 25000)))) + '€  et ' + str(formate(math.trunc((estimation + 25000)))) + '€.'))



