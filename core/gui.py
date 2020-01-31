import sys
import os
from tkinter import *


window=Tk()

window.title("Welcome to Deep Learning for Manufacturing (dlmfg)")
window.geometry('550x200')


def deploy_model():
	os.system('python -W ignore model_deployment.py')

B=Button(window,text="Run Model",command= deploy_model)
B.pack()
window.mainloop()