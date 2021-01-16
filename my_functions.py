
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Rectangle
from skimage import color
from skimage.transform import resize


#Function to easily view images
def show_image(image,title='Image',cmap_type='gray'):
    """Takes an Image's file path as an input
    and displays the image.
    Can specify a title if desired
    """
    plt.imshow(image,cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
    

#Function which loads images, converts them to grayscale and resizes them into an x by x image
def process_images(files,img_size):
    image_list = []
    """Takes a list of file paths leading to images and an integer as inputs, and converts
    the images into grayscale and resizes all the images to a standard square size.
    
    Returns a numpy array consisting of x by x arrays which represent each of the images.
    """ 
    for i in range(0,len(files)):
        img_data = plt.imread(files[i])
        img_data = color.rgb2gray(img_data)
        
        #Find the critical value
        thresh_otsu = threshold_otsu(img_data)
        #Apply thresholding
        sample_char = img_data <= thresh_otsu 
        
        img_data = resize(img_data,(img_size,img_size))
        image_list.append(img_data)
        
    #Converts the list into a numpy array so it can be used for neural networks
    image_list = np.array(image_list)
    
    return image_list


#Functions to easily view the total number of characters classes with less than
#or greater than a given number of observations

def total_less_than(number,df):
    """Takes an integer and a dataframe as inputs.
    Prints the number of classes in the dataframe that have a number
    of observations less than or equal
    to the integer.
    """
    number = int(number)
    amount = df['Count'] <= number
    print(f"\nNumber of characters with {number} or fewer observations: "+str(df[amount].shape[0]))
    print("This makes up: "+str(round(df[amount].shape[0]/df.shape[0],4)*100)+"% of the classes")


def total_greater_than(number,df):
    """Takes an integer and a dataframe as inputs.
    Prints the number of classes in the dataframe that have a number
    of observations greater than or equal
    to the integer.
    """
    number = int(number)
    amount = df['Count'] >= number
    print(f"\nNumber of characters with {number} or more observations: "+str(df[amount].shape[0]))
    print("This makes up: "+str(round(df[amount].shape[0]/df.shape[0],4)*100)+"% of the classes")



def total_equal_to(number,df):
    """Takes an integer and a dataframe as inputs.
    Prints the number of classes in the dataframe that have the same
    amount of observations as the integer.
    """
    number = int(number)
    amount = df['Count'] == number
    print(f"\nNumber of characters with {number} observations: "+str(df[amount].shape[0]))
    print("This makes up: "+str(round(df[amount].shape[0]/df.shape[0],3)*100)+"% of the classes")



#Function that uses coordinate information in the csv files associated with each book
#to draw boxes around all of the characters on a given page.
#It takes an image, the stem of the image file, and the coordinate information as inputs.

def find_characters(page_image,page_stem,coordinates):
    """Function which draws boundary boxes around each character on a given page.
    Takes 3 inputs. An image of a page (processed through plt.imread()),
    the page's file stem, and a csv file which contains the coordinates of
    the characters on the page.
    """

    plt.imshow(page_image)
    img_desc = plt.gca()

    #Retrives only the coordinates that are relevent to the page of interest
    page_coordinates = coordinates.loc[np.where(coordinates['Image']==page_stem)]

    for i in range(page_coordinates.shape[0]):
        #retrives the coordinates for each character on the page of interest
        x_min = int(coordinates['X'].loc[[i]])
        y_min = int(coordinates['Y'].loc[[i]])
        w = int(coordinates['Width'].loc[[i]])
        h = int(coordinates['Height'].loc[[i]])

        #Adds a green rectangle/boundary box to each of the characters and displays it on the image.
        img_desc.add_patch(Rectangle((x_min,y_min),w,h, fill=False, color = 'g',linewidth=1))

    plt.show()
