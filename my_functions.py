import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path


import keras.backend as K


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
    
    
#Function to calcualte f1 score for Keras Neural Networks
#Copied from keras's old source code
def neural_network_f1(y_true, y_pred): 
    
    #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


#Function to examine the amount of classes with a given f1 score    
def examine_f1(label,f1_scores,n,value=1.0,criteria='greater', details=False):
    """Takes three lists as inputs representing the labels in a test set,
    a list of f1_scores for each class, and a list of the number of 
    observations in each class.
    
    Users can specify an f1 value with the 'value' input.

    The function returns the amount of classes with an f1-score greater 
    than/equal to the 'value' input
    
    Changing greater to 'False' will return the number of classes with 
    an f1-score less than/equal to the 'value' input.
    
    Changing 'details' to true will return three lists with the specific
    class lables, their f1-score, and their number of observations.
    """
    
    total = 0
    unicode = []
    f1 = []
    support = []
    
    #If user specifies greater, find all classes with f1-scores greater than value
    if criteria == 'greater':
        for i in range(0,len(f1_scores)):
            if f1_scores[i] >= value:
                unicode.append(label[i])
                f1.append(f1_scores[i])
                support.append(n[i])
                total += 1
                
        #If user specifies details, return three lists with the specific classes 
        if details == True:
            print(f'Total classes with an f1 score greater than or equal to {value}: {total}')
            print(f'Average number of observations per class:{round(np.mean(support),2)}\n')
            return unicode, f1, support
        else:
            print(f'Total classes with an f1 score greater than or equal to {value}: {total}')
            return f'Average number of observations per class:{round(np.mean(support),2)}\n'
     
    
    #If user specifies less, find all classes with f1-scores less than value
    elif criteria == 'less':
        for i in range(0,len(f1_scores)):
            if f1_scores[i] <= value:
                unicode.append(label[i])
                f1.append(f1_scores[i])
                support.append(n[i])
                total += 1
        
        #If user specifies details, return three lists with the specific classes 
        if details == True:
            print(f'Total classes with an f1 score less than or equal to {value}: {total}')
            print(f'Average number of observations per class:{round(np.mean(support),2)}\n')
            return unicode, f1, support
        else:
            print(f'Total classes with an f1 score less than or equal to {value}: {total}')
            return f'Average number of observations per class:{round(np.mean(support),2)}\n'
        

    #If user specifies equal, find all classes with f1-scores equal to value
    elif criteria == 'equal':
        for i in range(0,len(f1_scores)):
            if f1_scores[i] == value:
                unicode.append(label[i])
                f1.append(f1_scores[i])
                support.append(n[i])
                total += 1
        
        #If user specifies details, return three lists with the specific classes 
        if details == True:
            print(f'Total classes with an f1 score equal to {value}: {total}')
            print(f'Average number of observations per class:{round(np.mean(support),2)}\n')
            return unicode, f1, support
        else:
            print(f'Total classes with an f1 score equal to {value}: {total}')
            return f'Average number of observations per class:{round(np.mean(support),2)}\n'


#Function to calculate the average f1 score based on the numeber of obsercations in a class.
def average_f1(support,support_list,f1_list):
    """This function calculates the average f1-score
    of classes with a set amount of observations.
    
    It takes 3 inputs, an integer specifying the number
    of observations, a list specifying the number of observations
    for each class, and a list of f1-scores.
    """
    av_f1 = []
    for i in range(0,len(support_list)):
        if support_list[i] == support:
            av_f1.append(f1_list[i])
            
    return round(np.mean(av_f1),2)

    
#Function to convert the x_train and x_test csv files into usable sets
def convert_xdata(dataset): 
    '''Takes a pandas dataframe as an input and converts it to a numpyarray of numpyarrays
    which contain binary values representing the images.'''
    dataset = np.array(dataset)
    newdata = []
    for i in range(0,dataset.shape[0]):
        data = dataset[i][0].split(" ")
        data = list(map(float,data))
        data = list(map(int,data))
        data = np.array(data)
        newdata.append(data)
        
    newdata = np.array(newdata)
    return newdata
        

#Function to calculate number of necessary augmentations and generate synthetic image data
def augment_classes(x_traindf, counted, number_of_samples=1):
    """
    Function to generate sample images such that each class has 100 observations each.
    The function takes 3 inputs, a dataframe containing the training data, a dictionary
    with each class and the number of observations in each class, and an integer representing
    how many samples to take when generating augmented data (default value is one).
    
    Function returns two lists representing the augmented images and the class labels associated
    with them.
    
    """

    to_augment_label = []
    number_of_augmentations = []

    N = number_of_samples;

    for key in counted.keys():
        #Count the number of augmentations required so each class has 100 observations
        to_augment_label.append(key)
        number_of_augmentations.append(100-counted.get(key))


    augmented_images = []
    augmented_labels = []
    
    for i in range(0,len(to_augment_label)):
        char_class = x_traindf[x_traindf["Unicode"] == to_augment_label[i]]
        for k in range(0,number_of_augmentations[i]):
            
            #Sample a random file from the character class 
            sample = char_class['Files'].sample(N,replace=True)
            sample = str(sample.values)
            file = Path(sample[2:len(sample)-2])
        
            #Generate augmented data from the sampled image
            augmented_images.append(generate_data(file,n=N))

            #Make sure that the augmentations have labels associated with them
            augmented_labels.append(to_augment_label[i])
            
    return augmented_images, augmented_labels

    
    

#Function to generate synthetic data samples
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
    datagen = ImageDataGenerator(zoom_range=[0.9,1.0],brightness_range=[0.5,1.0],
                                 rotation_range=10,height_shift_range=0.1,width_shift_range=[-5,5])
    
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
    img_data = resize(img_data,(64,64))
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


    
#Function to check if a list contains duplicate values
def checkDuplicates(list_of_elems):
    ''' Check if given list contains any duplicates '''
    if len(list_of_elems) == len(set(list_of_elems)):
        return False
    else:
        return True

    
#Function to count the number of unique classes in an arrays and return the count of each class   
def count_characters(characters,return_dict=True):
    
    """Function takes an input of class labels as a numpy array 
    and returns a dictionary of each class with the amount of unique 
    occurances of that class.
    
    Users can specify if they would like the output as a dictionary or
    as two seperate arrays representing the unique characters and their counts.
    """
    
    unique_char, counts = np.unique(characters,return_counts=True)
    
    if return_dict == True:
        counted = dict(zip(unique_char,counts))
        return counted
    
    elif return_dict == False:
        return unique_char, counts

    

#Function to produce history plot of a neural network's performance across epochs
def show_history(model_history,metric,file_name):
    """
    Function which generates a neural network's loss or accuracy performance across
    epochs. It takes three inputs, a 'history' object from Keras callbacks, the metric
    name (accuracy or loss)
    """
    #Produce a plot of a model's accuracy aross epochs
    if metric == 'accuracy':
        plt.plot(model_history.history['accuracy'])
        plt.plot(model_history.history['val_accuracy'])
        plt.title(file_name)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        file_name = str(file_name) + '.png'
        plt.savefig(file_name)
        plt.show()
        return
        
    elif metric == 'loss':
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title(file_name)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        file_name = str(file_name) + '.png'
        plt.savefig(file_name)
        plt.show()
        return
    else:
        print('Please specify either loss or accuracy (using single quotes)')
    

