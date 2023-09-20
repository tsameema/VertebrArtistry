import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import cv2, tensorflow
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras import Model

class VertebrasLandmarking():
   
    def __init__(self, model_path):
        """
            Initializes the VertebrasLandmarking class.
            
            Args:
                model_path (str): The path to the pre-trained model file.
        """
        self.model_path = model_path

    def image_processing(self, imagefile):
        """
            Preprocesses the input image for landmark prediction.
            
            Args:
                imagefile (str): The path to the input image file.
                
            Returns:
                numpy.ndarray: The preprocessed image as a NumPy array.
        """
        loadimage=cv2.imread(imagefile)
        #converting rgb to gray
        greyimage=cv2.cvtColor(loadimage, cv2.COLOR_BGR2GRAY)
        #equalizing intensities
        histogramimage=cv2.equalizeHist(greyimage)
        #removing noise
        remove_image_noise= cv2.fastNlMeansDenoising(histogramimage, None, 10, 7, 21) 
        #Normalizing
        normalizedimage=(remove_image_noise - np.min(remove_image_noise)) / (np.max(remove_image_noise) - np.min(remove_image_noise))
        return normalizedimage
    
    def upload_model(self):
        """
            Loads the pre-trained model.
            
            Returns:
                tensorflow.keras.Model: The loaded pre-trained model.
        """
        model = tensorflow.keras.models.load_model(self.model_path)
        print('MODEL LOADED')
        return model
    
    def upload_file(self):
        """
            Upload the test files
            
            Returns:
                None
        """
        f_types = [('Jpg Files', '*.jpg'),
        ('PNG Files','*.png')]   # type of files to select 
        filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
        for imagefile in filename:
            self.process_predict_testfile(imagefile)

    def process_predict_testfile(self, imagefile):
        """
            Preprocesses the input image and predicts landmarks using the loaded model.
            
            Args:
                imagefile (str): The path to the input image file.
                
            Returns:
                None
        """
        image_array=self.image_processing(imagefile)
        print(image_array.shape)
        img_expand_dim=np.expand_dims(image_array, axis=0)
        print(img_expand_dim.shape)
        img_expand_dimtest=np.repeat(np.expand_dims(img_expand_dim, axis=3), 3, axis=3)
        print(img_expand_dimtest.shape)

        ## **TESTING THE TRAINED MODEL UPON TEST EXAMPLES
        ypred=self.upload_model().predict(img_expand_dimtest).astype(int)

        self.display_result(ypred, image_array)

    def display_result(self, ypred, image_array):  
        """
        Plots the input image with predicted landmarks.
        
        Args:
            imagefile (str): The path to the input image file.
        """

        xP, yP=ypred[0,:][:,0], ypred[0,:][:,1]
        # Use TkAgg
        matplotlib.use("TkAgg")
        # Create a figure of specific size      
        fig1 = plt.figure(figsize=(5,5), dpi=100)
        # Define the points for plotting the figure
        plot = fig1.add_subplot(1, 1, 1)
        plot.imshow(image_array)
        plot.scatter(xP,yP, color='red', s=10)
        # Add a canvas widget to associate the figure with canvas
        canvas = FigureCanvasTkAgg(fig1, my_window)
        canvas.get_tk_widget().grid(row=0, column=0)         


if __name__ == "__main__":

    model_path = "Model_denoise_hist_e300_resnet101_b8.h5"
    landmark = VertebrasLandmarking(model_path)

    my_window = tk.Tk()
    my_window.geometry("500x500")  # Size of the window 
    my_window.title('VERTEBRAS LANDMARKING')
    my_font1=('times', 18, 'bold')
    l1 = tk.Label(
        my_window,text='Upload Images',width=30,font=my_font1
        )  
    l1.grid(row=1,column=1,columnspan=4)
    b1 = tk.Button(
        my_window, text='Upload', width=20,command = lambda:landmark.upload_file()
        )
    b1.grid(row=2,column=1,columnspan=4)

    my_window.mainloop()  # Keep the window open
