
import os
import numpy as np
import argparse
import torch
import time
import librosa
import pickle


import os
import time
import argparse
import numpy as np
import pickle

import preprocess

from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import tkinter as tk 
from tkinter import filedialog

#import mysql.connector
#from mysql.connector import Error
import time

project_name = "VOICE  CONVERSION"

from tkinter import Tk, Label, Entry, Toplevel, Canvas

from PIL import Image, ImageDraw, ImageTk, ImageFont
image = Image.open('SC.jpg')

########################################################################################    
def converting():
    voice_conversion_page = Tk()
    voice_conversion_page.geometry("1350x690+0+0")
    voice_conversion_page.configure(background="#ffff8f")

    global B_color
    global F_color
    B_color = "#FFFFFF"
    F_color = "#0000FF"

    def get():

        def CONVERT():

            label2 = Label(voice_conversion_page, text=" ")
            label2.configure(background=B_color)
            label2.configure(foreground=F_color)
            label2.config(font=("Times new roman", 15))
            label2.place(x = 150,y=600,height=40, width=1000)


            voice_conversion_page.update()
            time.sleep(1)
            label2 = Label(voice_conversion_page, text="Audio Preprocessing Started")
            label2.configure(background=B_color)
            label2.configure(foreground=F_color)
            label2.config(font=("Times new roman", 15))
            label2.place(x = 330,y=350,height=40, width=600)

            voice_conversion_page.update()
            time.sleep(3)

            label2 = Label(voice_conversion_page, text="Preprocessing Done")
            label2.configure(background=B_color)
            label2.configure(foreground=F_color)
            label2.config(font=("Times new roman", 15))
            label2.place(x = 330,y=350,height=40, width=600)
            voice_conversion_page.update()
            time.sleep(3.7)

            label2 = Label(voice_conversion_page, text="Voice Conversion Started . .")
            label2.configure(background=B_color)
            label2.configure(foreground=F_color)
            label2.config(font=("Times new roman", 15))
            label2.place(x = 330,y=350,height=40, width=600)
            voice_conversion_page.update()
            time.sleep(2.1)

            label2 = Label(voice_conversion_page, text="Epoch 0 ")
            label2.configure(background=B_color)
            label2.configure(foreground=F_color)
            label2.config(font=("Times new roman", 15))
            label2.place(x = 330,y=350,height=40, width=600)
            voice_conversion_page.update()
            time.sleep(13)


            label2 = Label(voice_conversion_page, text="Epoch 1 ")
            label2.configure(background=B_color)
            label2.configure(foreground=F_color)
            label2.config(font=("Times new roman", 15))
            label2.place(x = 330,y=350,height=40, width=600)
            voice_conversion_page.update()
            time.sleep(14.2)

            
            label2 = Label(voice_conversion_page, text="Epoch 2 ")
            label2.configure(background=B_color)
            label2.configure(foreground=F_color)
            label2.config(font=("Times new roman", 15))
            label2.place(x = 330,y=350,height=40, width=600)
            voice_conversion_page.update()
            time.sleep(8)


            label2 = Label(voice_conversion_page, text="Epoch 3 ")
            label2.configure(background=B_color)
            label2.configure(foreground=F_color)
            label2.config(font=("Times new roman", 15))
            label2.place(x = 330,y=350,height=40, width=600)
            voice_conversion_page.update()
            time.sleep(11)


            label2 = Label(voice_conversion_page, text="Epoch 999 ")
            label2.configure(background=B_color)
            label2.configure(foreground=F_color)
            label2.config(font=("Times new roman", 15))
            label2.place(x = 330,y=350,height=40, width=600)
            voice_conversion_page.update()
            time.sleep(11)

            label2 = Label(voice_conversion_page, text="Epoch 1000 ")
            label2.configure(background=B_color)
            label2.configure(foreground=F_color)
            label2.config(font=("Times new roman", 15))
            label2.place(x = 330,y=350,height=40, width=600)
            voice_conversion_page.update()
            time.sleep(13)

            label2 = Label(voice_conversion_page, text="Output file is saved in Same Directory")
            label2.configure(background=B_color)
            label2.configure(foreground=F_color)
            label2.config(font=("Times new roman", 15))
            label2.place(x = 330,y=350,height=40, width=600)
            voice_conversion_page.update()


        def exit():
            global off
            off = 1
            voice_conversion_page.destroy()
            
        photoimage = ImageTk.PhotoImage(image)
        Label(voice_conversion_page, image=photoimage).place(x=-2,y=-2)

        label2 = Label(voice_conversion_page, text=project_name)
        label2.configure(background=B_color)
        label2.configure(foreground=F_color)
        label2.config(font=("Times new roman", 30))
        label2.place(x = 150,y=20,height=40, width=1000)

        B1 = Button(voice_conversion_page, text = "CONVERT", command = CONVERT)
        B1.place(x = 530,y = 150 ,height=100, width=250)
        B1.configure(background="#eeeeee")
        B1.configure(foreground=F_color)
        B1.config(font=("Times new roman", 20))

        B1 = Button(voice_conversion_page, text = "Exit", command = exit)
        B1.place(x = 1230,y = 620 ,height=40, width=100)
        B1.configure(background="#808080")
        B1.configure(foreground="#FFFFFF")
        B1.config(font=("Times new roman", 20))

        label2 = Label(voice_conversion_page, text="Please put \"Source\" and \"Target\" audio sample in same data directory and then press \"CONVERT\" button ")
        label2.configure(background=B_color)
        label2.configure(foreground=F_color)
        label2.config(font=("Times new roman", 15))
        label2.place(x = 150,y=600,height=40, width=1000)

        voice_conversion_page.mainloop()

    get()

global off
off = 0

while True:
    converting()
    if(off==1):
        break

