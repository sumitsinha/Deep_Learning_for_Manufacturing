#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 19:34:58 2020

@author: sinha_s
"""
#%%
import pandas as pd
import numpy as np
#import cufflinks as cf
import plotly as py
from plotly.offline import iplot
import plotly.express as px
import plotly.graph_objects as go


user_preds = pd.read_csv("user_preds.csv")
user_inputs= pd.read_csv("user_inputs.csv")

#print(user_inputs)
user_names=user_inputs.iloc[:,0:1].values
errors=user_inputs.iloc[:,1:7].values-user_preds.values
errors=np.absolute(errors)
mae=errors.mean(axis=1)  
#print(user_names)

leaderboard_final= pd.DataFrame({'User Name': user_names[:, 0], 'Mean Absolute Error': mae})
print(leaderboard_final)
#xl = pd.ExcelFile("../leaderboard/leaderboard.xlsx")
#leaderboard_final = xl.parse("user_errors")
leaderboard_display=leaderboard_final.groupby(["User Name"],sort='true')['Mean Absolute Error'].max().reset_index()
# #%%
leaderboard_display["text"] = leaderboard_display["User Name"]+" - "+leaderboard_display["Mean Absolute Error"].map(str)+" mm" 

fig = px.scatter(leaderboard_display, text="text",y="Mean Absolute Error",x="User Name",size="Mean Absolute Error")
fig.update_traces(textposition='top center')

fig.update_layout(
    title_text='Engineering Game - Leaderboard',
)
#fig=leaderboard_final.iplot(kind='scatter',text="User Name",y="Mean Absolute Error",asFigure=True,mode="markers")
py.offline.plot(fig,filename="leaderboard.html")




