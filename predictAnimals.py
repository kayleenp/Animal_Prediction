# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 22:07:33 2019

@author: User
"""

import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.graph_objs as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
import os
import pandas as pd
import dash_core_components as dcc
import warnings
warnings.filterwarnings("ignore")

##CREATING DATAFRAME ##
dataframe = pd.read_csv("C:/Users/User/Downloads/zoo_.csv")
df = dataframe

##DROP ANIMAL_NAME BECAUSE WE WANT TO PREDICT THAT
features = df.drop("animal_name", axis =1)
target = df["animal_name"]

#SPLIT TRAIN FEATURES, TEST FEATURES, TRAIN TARGETS, AND TEST TARGETS
train_features, test_features, train_targets, test_targets = \
        train_test_split(features, target, train_size=0.75)
        

#USING DECISION TREE FOR CLASSIFIER AND PREDICTION       
tree = DecisionTreeClassifier(criterion="entropy", max_depth=5)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#USER INTERFACE USING DASH
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([ html.Div([html.H6("Hair"),
  dcc.Slider(id = 'input-hair', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = {'width':70}), 

  html.Div([html.H6("Feathers"),
  dcc.Slider(id = 'input-feathers', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':70}), 

     html.Div([html.H6("Eggs"),
  dcc.Slider(id = 'input-eggs', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':70}), 


     html.Div([html.H6("Milk"),
  dcc.Slider(id = 'input-milk', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = { 'width':70 ,'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30}), 
    
     html.Div([html.H6("Airborne"),
  dcc.Slider(id = 'input-airborne', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ], style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':70}), 

     html.Div([html.H6("Aquatic"),
  dcc.Slider(id = 'input-aquatic', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ], style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':70}), 
    
     html.Div([html.H6("Predator"),
  dcc.Slider(id = 'input-predator', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':70}), 

     html.Div([html.H6("Toothed"),
  dcc.Slider(id = 'input-toothed', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':70}), 

     html.Div([html.H6("Backbone"),
  dcc.Slider(id = 'input-backbone', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':70}), 

     html.Div([html.H6("Breathes"),
  dcc.Slider(id = 'input-breathes', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':70}), 
    
     html.Div([html.H6("Venomous"),
  dcc.Slider(id = 'input-venomous', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':70}), 

     html.Div([html.H6("Fins"),
  dcc.Slider(id = 'input-fins', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':70}), 

     html.Div([html.H6("Legs"),
  dcc.Slider(id = 'input-legs', 
    min=0,
    max=8,
    marks={
            0 : '0',
            1 : '1', 
            2 : '2' , 
            3 : '3' ,
            4 : '4',
            5 : '5',
            6 : '6',
            7 : '7',
            8 : '8'},
    value=4
   ) ],  style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':170}), 

     html.Div([html.H6("Tail"),
  dcc.Slider(id = 'input-tail', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':70} ), 
     html.Div([html.H6("Domestic"),
  dcc.Slider(id = 'input-domestic', 
    min=0,
    max=1,
    marks={
            1 : 'YES', 
            0 : 'NO' },
    value=1
   ) ],  style = { 'width':70} ), 

  html.Div([html.H5("Class Type"), html.H6("1. Mammal, 2. Birds, 3. Fish, 4. Reptile , 5. Amphibian , 6. Arthropods, 7. Crustaceans \n  "),
  dcc.Slider(id = 'input-class-type', 
    min=1,
    max=7,
    marks={   
            1 : '1', 
            2 : '2' , 
            3 : '3' ,
            4 : '4',
            5 : '5',
            6 : '6',
            7 : '7'
            },
    value=7
   ) ],  style = {'margin-left':10, 'margin-bottom' : 30, 'margin-top' : 30, 'width':200} ),
    
    html.Div(id = 'output-animal-hair'),
    html.Div(dcc.Input(id='input-box', type='text')),
    html.Button('Submit', id='button'),
    html.Div(id='output-container-button',
             children='Enter a value and press submit'),
    html.Button('Refresh', id='newPredButton'),
    html.Div(id='output-refresh')], style={'columnCount': 4})
    
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

    



@app.callback(Output('output-animal-hair',  'children'), 
              [Input('input-hair','value'),
               Input('input-feathers','value'),
               Input('input-eggs','value'),
               Input('input-milk','value'),
               Input('input-airborne','value'),
               Input('input-aquatic','value'),
               Input('input-predator','value'),
               Input('input-toothed','value'),
               Input('input-backbone','value'),
               Input('input-breathes','value'),
               Input('input-venomous','value'),
               Input('input-fins','value'), 
               Input('input-tail','value'),
               Input('input-domestic','value'), 
               Input('input-class-type', 'value'), 
               Input('input-legs' , 'value')])
         
def return_value (input_hair, input_feathers, input_eggs, input_milk, input_airborne, 
                  input_aquatic, input_predator, input_toothed, input_backbone, 
                  input_breathes, input_venomous, input_fins, input_tail, input_domestic, input_class_type, input_legs):
        ##INPUT THE FEATURES##
        features = {
                    "hair": input_hair, 
                    "feathers": input_feathers,
                    "eggs": input_eggs,
                    "milk": input_milk,
                    "airborne": input_airborne,
                    "aquatic": input_aquatic,
                    "predator": input_predator,
                    "toothed": input_toothed,
                    "backbone": input_backbone,
                    "breathes": input_breathes,
                    "venomous": input_venomous,
                    "fins": input_fins,
                    "legs": input_legs,
                    "tail": input_tail,
                    "domestic": input_domestic,
                    "class_type" : input_class_type
                    }

        features = pd.DataFrame([features], columns=train_features.columns)

        trees = tree.fit(train_features, train_targets)
        prediction= trees.predict(features) 
        
        return "PREDICTION: {} ".format(prediction)
    
    
    
@app.callback(Output('output-refresh', 'children'),
              [Input('newPredButton', 'n_clicks')], 
              [State('input-hair','value'),
               State('input-feathers','value'),
               State('input-eggs','value'),
               State('input-milk','value'),
               State('input-airborne','value'),
               State('input-aquatic','value'),
               State('input-predator','value'),
               State('input-toothed','value'),
               State('input-backbone','value'),
               State('input-breathes','value'),
               State('input-venomous','value'),
               State('input-fins','value'), 
               State('input-tail','value'),
               State('input-domestic','value'), 
               State('input-class-type' , 'value'), 
               State('input-legs', 'value')]) 


#TO FIND OTHER PREDICTION
def refresh_prediction(newPredButton, input_hair, input_feathers, input_eggs, input_milk, input_airborne, input_aquatic, input_predator, input_toothed, 
                       input_backbone, input_breathes, input_venomous, input_fins, input_tail, input_domestic, input_class_type, input_legs): 
                           
         
         features = {
                    "hair": input_hair, 
                    "feathers": input_feathers,
                    "eggs": input_eggs,
                    "milk": input_milk,
                    "airborne": input_airborne,
                    "aquatic": input_aquatic,
                    "predator": input_predator,
                    "toothed": input_toothed,
                    "backbone": input_backbone,
                    "breathes": input_breathes,
                    "venomous": input_venomous,
                    "fins": input_fins,
                    "legs": input_legs,
                    "tail": input_tail,
                    "domestic": input_domestic,
                    "class_type" : input_class_type
                    }
         
         ##FITTING AND PREDICTION
         features = pd.DataFrame([features], columns=train_features.columns)
         trees = tree.fit(train_features, train_targets)
         prediction= trees.predict(features) 
         
         return "OTHER PREDICTION {}".format(prediction)   
    
@app.callback(Output('output-container-button', 'children'),
              [Input('button', 'n_clicks'), 
               Input('input-hair','value'),
               Input('input-feathers','value'),
               Input('input-eggs','value'),
               Input('input-milk','value'),
               Input('input-airborne','value'),
               Input('input-aquatic','value'),
               Input('input-predator','value'),
               Input('input-toothed','value'),
               Input('input-backbone','value'),
               Input('input-breathes','value'),
               Input('input-venomous','value'),
               Input('input-fins','value'), 
               Input('input-tail','value'),
               Input('input-domestic','value'),
               Input('input-class-type', 'value'), 
               Input('input-legs' , 'value'),
               Input('input-box', 'value')])

##UPDATE USER INPUT DATA INTO THE CSV FILE  
def update_output(button, input_hair, input_feathers, input_eggs, input_milk, input_airborne, 
                  input_aquatic, input_predator, input_toothed, input_backbone, 
                  input_breathes, input_venomous, input_fins, input_tail, input_domestic,input_class_type, input_legs, input_box):
         
      
       
        newAnimal = dataframe.append( {"animal_name" : input_box,
                    "hair": input_hair, 
                    "feathers": input_feathers,
                    "eggs": input_eggs,
                    "milk": input_milk,
                    "airborne": input_airborne,
                    "aquatic": input_aquatic,
                    "predator": input_predator,
                    "toothed": input_toothed,
                    "backbone": input_backbone,
                    "breathes": input_breathes,
                    "venomous": input_venomous,
                    "fins": input_fins,
                    "legs": input_legs,
                    "tail": input_tail,
                    "domestic": input_domestic,
                    "class_type" : input_class_type}, ignore_index = True) 
        
        newAnimal.to_csv("C:/Users/User/Downloads/zoo_.csv", index = False)
        return "{} has been updated in training data ".format(input_box)
         
      
     
if __name__ == '__main__':
    app.run_server(debug=True)


    


#2. Choose Decision Trees
