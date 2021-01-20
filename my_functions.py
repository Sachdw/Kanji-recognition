import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp


from tqdm import tqdm
from matplotlib.patches import Rectangle
from skimage.filters import threshold_otsu
from skimage import color
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator



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
        
    
def generate_data(file, view="no", n = 9):
    """Takes an Image file path as an input and augments the image
    randomly to create n samples (default is 9). The user can specify if they wish to view the
    samples by specifiying "view = yes", and set the amount of samples to generate 
    by setting "n = number of samples".
    """
    new_images = []
    
    # load the image to be augmented
    img_data = plt.imread(file)

    # expand dimension to one sample
    samples = np.expand_dims(img_data, 0)
    
    # create image data augmentation generator
    datagen = ImageDataGenerator(zoom_range=[0.7,1.0],brightness_range=[0.4,1.0],
                                 rotation_range=[-10,10],height_shift_range=0.1,width_shift_range=[-10,10])
    
    # prepare iterator
    iterator = datagen.flow(samples, batch_size=1)
    
    # generate samples and plot
    for i in range(n):
        
        #generate batch of images
        batch = iterator.next()
        # convert to unsigned integers for viewing
        new_image = batch[0].astype('uint8')
        #Add numpy array to list of augmented images
        new_images.append(new_image)
        
        #Create a plot of the augmented images if a user specifies
        if(view == 'yes'):
            #define subplot
            plt.subplot(330 + 1 + i)
            #plot raw pixel data
            plt.imshow(new_image)

    # show the figure if user specifies
    if(view == "yes"):
        plt.show()
    
    #Return list of arrays
    return new_images

 

#Function which loads images, converts them to grayscale and resizes them into an x by x image
def process_images(file):
    """Takes a file path leading to an image or a numpy array as an input, and converts
    the images into grayscale and resizes all the images to a standard square size.
    
    Returns a numpy array consisting of 65 by 65 arrays which represent each of the images.
    """ 
    
    img_data = file
    
    #If the input is a file path
    if(type(file) != np.ndarray):
        #Read image file
        img_data = plt.imread(file)
        
    #Resize image
    img_data = resize(img_data,(65,65))
    #Convert it to grayscale
    img_data = color.rgb2gray(img_data)
    #Find the critical value for threshoulding
    thresh_otsu = threshold_otsu(img_data)
    #Apply thresholding
    img_data = img_data <= thresh_otsu 

    return img_data
 
    

#Function to parallelize image processing
def parallelize_process(img_list):
    """Takes a list of file paths leading to images as an input and uses parallel processing to 
    process each of the images. Returns a list of numpy arrays representing the processed images.
    """
    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    print(f"Starting the image processing with {cores} processing units.")
    results = pool.map(process_images,tqdm(img_list))
    print("Closing the pool...")
    pool.close()
    print("Joining the pool...")
    pool.join()
    return results


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
    num_of_classes = df[amount].shape[0]
    print(f"\nNumber of characters with {number} observations: {num_of_classes}")
    print("This makes up: "+str(round(df[amount].shape[0]/df.shape[0],4)*100)+"% of the classes")
    
    
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
