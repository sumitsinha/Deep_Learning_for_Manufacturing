#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 19:34:58 2020

@author: sinha_s
"""
#%%
import pandas as pd
import cufflinks as cf
import plotly as py
from plotly.offline import iplot
import plotly.express as px
import plotly.graph_objects as go

xl = pd.ExcelFile("../leaderboard/leaderboard.xlsx")
leaderboard_final = xl.parse("user_errors")
leaderboard_display=leaderboard_final.groupby(["User Name"],sort='true')['Mean Absolute Error'].max().reset_index()
#%%
leaderboard_display["text"] = leaderboard_display["User Name"]+" - "+leaderboard_display["Mean Absolute Error"].map(str)+" mm" 

fig = px.scatter(leaderboard_display, text="text",y="Mean Absolute Error",x="User Name",size="Mean Absolute Error")
fig.update_traces(textposition='top center')

fig.update_layout(
    title_text='Engineering Game - Leaderboard',
)
#fig=leaderboard_final.iplot(kind='scatter',text="User Name",y="Mean Absolute Error",asFigure=True,mode="markers")
py.offline.plot(fig,filename="leaderboard.html")




