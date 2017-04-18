# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:36:36 2017

@author: iiss
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2


from imgaug import augmenters as iaa


# minimum steering value below which to  ignore csv
LIMIT = 0.0

current_dir = "data/IMG/"
csv_file = "data/driving_log.csv"

    
    
# Equlaize the intensity in RGB image

def hist_equal(img_path):
    
    """
    Takes in the path of an image file and 
    perform histogram equalization using YUV space
    """
    # Get the BGR cv2 image
    img_in = cv2.imread(img_path)
    # Convert o YUV
    img_yuv = cv2.cvtColor(img_in, cv2.COLOR_BGR2YUV)
    
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    
    # convert the YUV image back to RGB format
    img_out = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    return img_out



def get_data_generator(csv_data, batch_size=64,height = 160, width = 320):
    
    
    """
    Generator to create training data
    
    Arguments
    csv_data: A list where each row is a line from csvfile
    batch_size:  images per batch
    height : Height of input image
    width : Width of input image
    """
    
    N = len(csv_data)           # total data
    batches_per_epoch = N // batch_size   # number of bacthes
    
    i = 0
    np.random.shuffle(csv_data)
       
    # initialize batch as floats   
    X_batch = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size,), dtype=np.float32)
   
    while(True):
        #get start and end point
        start = i*batch_size
        end = start+batch_size - 1

        # initialize your batch data
        j = 0

        # slice a `batch_size` sized chunk from the csv_data
        # and generate augmented data for each row in the chunk on the fly
        for index, row in enumerate(csv_data[start:end]):

           # generate an augmented image from a true image
            X_batch[j], y_batch[j] = get_augmented_row(row)
            j += 1

        i += 1
        
        # If all data is read, restart from beginning
        if i == batches_per_epoch - 1:
            i = 0
            np.random.shuffle(csv_data)
        yield X_batch, y_batch


def get_augmented_row(row):
    
    """
    A function which return an augmented image
    and the corresponding steering value 
    
    Argument :  an input line from csvfile
    """
    
    comp_val = 0.2385                   # compensation value for left and right caemras
    
    leaning = np.random.choice([0,1,2])  # 3 camera options
    
    # get one of the three possible cameras
    source_address = row[leaning]
    img_name = source_address.split('/')[-1]

    # getting the right image path
    img_path = current_dir + img_name

     # Equlaize the image
    eq_img = hist_equal(img_path)
   
    #perform random image augmentation techniques
    img = aug_img(eq_img)
    
     # Append Steering value
    steering = np.float32(row[3])
    # Change steering value based on camera
    if leaning == 1:
        steering += comp_val # or any other constant value which works well
    elif leaning == 2:
        steering -= comp_val

    # Flip image and steering value if the last column is TRUE
    if (row[-1]):
        img = np.fliplr(img)
        steering = steering * -1
 
    
    return img, steering


def read_csvfile():
    csv_lines = []
    # Read CSV file
    with open(csv_file) as csvfile: 
        next(csvfile)               #skip first line
        reader = csv.reader(csvfile)
        for line in reader:
            csv_lines.append(line)    # Appending line from csv file to the list
 

    # Get a list of indices to be randomly removed
    drop = []
    for ind,row in enumerate(csv_lines)  :
        if (  np.abs(float(row[3])) < LIMIT ):     # Get all throttle values below LIMIT
            drop.append(ind)
        
    np.random.shuffle(drop)
    
    a = len(drop)
    # remove 50% of all indices previously collected 
    drop[int(0.5*a):-1] = []
    z = sorted(drop, reverse=True)


    # remove those indices  from above from csv file
    for q in range((len(z))):
        del csv_lines[q]
    

    #  Copying the csvread list and adding a new column s
    temp_list =  []
    for ind,row in enumerate(csv_lines) :
         temp_list.append(row)  
         temp_list[-1].append(False) 

    # Copying the list
    new_list = list(temp_list) 
    
    # Change the last column to  True, indicating that image be inverted
    # in generator
    for ind,row in enumerate(new_list)        : 
            row[-1] = True

    # Final list of images to be used
    final_list = temp_list + new_list       
    print(np.shape(new_list))
    np.random.shuffle(final_list)
  
    
    return final_list










def aug_img(image):
    """
    The function takes input an image and performs, gaussian blur, 
    adding brightness, hsarpening, and contrast improvement on the images in a random fashion. 
    This function uses the imageaug libray
    
    Argument : Input image
    """


     # changin color space of the images
    img_bgr = cv2.cvtColor(np.squeeze(image),cv2.COLOR_RGB2BGR)
    
    
    #  Add gaussian blur
    gauss_blur = iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.2), per_channel=0.5)
    img_aug01 = gauss_blur.augment_image(img_bgr) 
    
    # add random brightness, and choose channels randomly
    add_brightness = iaa.Add((-10, 10), per_channel=0.5)
    img_aug1 = add_brightness.augment_image(img_aug01)
    
    # Improve sharpness 
    add_sharp = iaa.Sharpen(alpha=(0, 1.0), strength=(0.75, 1.5))
    img_aug2 = add_sharp.augment_image(img_aug1)
    ##
    
    ### Performing Contrast normalization
    rot = iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)
    img_aug = rot.augment_image(img_aug2)
    

    # Change colorspace to RGB    
    dest = cv2.cvtColor(img_aug,cv2.COLOR_BGR2RGB)
    
    return dest