from main import *

import tkinter as tk
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = Tk()
root.title("PGM/PPM Image Editor")
root.geometry("500x500")

pgm_ops = PgmOperations()
ppm_ops = PpmOperations()

pgm = None
ppm = None
canvas = None
prev_pgm = None
prev_ppm = None

def show():
    global canvas
    if canvas:
        canvas.get_tk_widget().destroy()

    figure = plt.Figure(figsize=(5, 4), dpi=100)
    image = figure.add_subplot(111)
    if pgm:
        image.imshow(pgm.image_mat, cmap='gray')
    elif ppm:
        image.imshow(ppm.image_mat)
    canvas = FigureCanvasTkAgg(figure, root)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

def browseFiles():
    global pgm
    global ppm
    path = filedialog.askopenfilename(initialdir = "~/",
                                          title = "Select a File",
                                          filetypes = (("PGM Files",
                                                        "*.pgm"),
                                                       ("PPM Files",
                                                        "*.ppm")))
    if path.split('.')[-1] == 'pgm':
        ppm=None
        pgm=pgm_ops.read(path)
        show()
    elif path.split('.')[-1] == 'ppm':
        pgm=None
        print(path)
        ppm=ppm_ops.read(path)  
        show()   

def save_file():
    pass
def mean():
    global pgm, prev_pgm
    if pgm:
        prev_pgm = pgm
        pgm = pgm_ops.mean_filter(pgm, 3)
        show()
def noise():
    global pgm, prev_pgm
    if pgm:
        prev_pgm = pgm
        pgm=pgm_ops.noise(pgm)    
        show()
def median():
    global pgm, prev_pgm
    if pgm:
        prev_pgm = pgm
        pgm = pgm_ops.median_filter(pgm, 3)
        show()
def undo():
    global pgm, prev_pgm, prev_ppm, ppm
    if prev_pgm and pgm:
        pgm = prev_pgm
        prev_pgm = None
        show()
    if prev_ppm and ppm:
        ppm = prev_ppm
        prev_ppm = None
        show()
    
def highpassing():
    global pgm, prev_pgm
    if pgm:
        prev_pgm = pgm
        pgm = pgm_ops.highpassing_filter(pgm)
        show()


def threshhold(cond):
    global ppm, prev_ppm
    if ppm:
        histogram_red = ppm_ops.histogram(ppm , 0)
        histogram_green = ppm_ops.histogram(ppm, 1)
        histogram_blue = ppm_ops.histogram(ppm, 2)
        otsu_red = ppm_ops.otsu_thresholding(histogram_red)
        otsu_green = ppm_ops.otsu_thresholding(histogram_green)
        otsu_blue = ppm_ops.otsu_thresholding(histogram_blue)
        prev_ppm = ppm
        ppm = ppm_ops.threshhold(ppm, (otsu_red, otsu_green, otsu_blue), cond)
        show()
def threshhold_and():
    threshhold('AND')
def threshhold_or():
    threshhold('OR')

def erode():
    global ppm, prev_ppm
    if ppm:
        prev_ppm = ppm
        ppm = ppm_ops.erosion(ppm, 3)
        show()
def delate():
    global ppm, prev_ppm
    if ppm:
        prev_ppm = ppm
        ppm = ppm_ops.dilatation(ppm, 3)
        show()
def closing():
    global ppm, prev_ppm
    if ppm:
        prev_ppm = ppm
        ppm = ppm_ops.fermeture(ppm, 3)
        show()
def opening():
    global ppm, prev_ppm
    if ppm:
        prev_ppm = ppm
        ppm = ppm_ops.fermeture(ppm, 3)
        show()
def histogram():
    global ppm, pgm
    if ppm:
        ppm_ops.draw_histogram(ppm)
    if pgm:
        pgm_ops.draw_histogram(pgm_ops.histogram(pgm))
def histogram_cum():
    global pgm, ppm
    if ppm:
        ppm_ops.draw_histogram_cumule(ppm)
    if pgm:
        pgm_ops.draw_histogram(pgm_ops.histogram_cumul(pgm))
def histogram_eg():
    global pgm,ppm
    if ppm:
        ppm_ops.draw_histogram_egalise(ppm)
    if pgm:
        pgm_ops.draw_histogram(pgm_ops.histogram_egalise(pgm))
menu = Menu(root)
root.config(menu=menu)
file_menu = Menu(menu)
menu.add_cascade(label='File', menu=file_menu)
file_menu.add_command(label='Open...', command=browseFiles)
file_menu.add_command(label='Save')
file_menu.add_separator()
file_menu.add_command(label='Exit', command=root.quit)
edit_menu = Menu(menu)
menu.add_cascade(label='Edit', menu=edit_menu)
edit_menu.add_command(label='Undo', command = undo)

edit_menu.add_command(label='Noise', command = noise)
edit_menu.add_command(label='Mean Filter', command = mean)
edit_menu.add_command(label='Median Filter', command = median)
edit_menu.add_command(label='High passing Filter', command=highpassing)
edit_menu.add_command(label='threshhold and', command = threshhold_and)
edit_menu.add_command(label='threshhold or', command = threshhold_or)

edit_menu.add_command(label='Erode', command=erode)
edit_menu.add_command(label='Dilate', command = delate)
edit_menu.add_command(label='Closing', command = closing)
edit_menu.add_command(label='Opening', command = opening)
display_menu = Menu(menu)
menu.add_cascade(label='Display', menu=display_menu)
display_menu.add_command(label='histogram', command = histogram)
display_menu.add_command(label='histogram cumulé', command = histogram_cum)
display_menu.add_command(label='histogram égalisé', command = histogram_eg)



mainloop()

