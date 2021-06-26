from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START CODE HERE ***

    #Reshape the image to 2-d
    w,h,d = image.shape
    x = image.reshape((w * h, d))  

    #Find out the total number of image points
    #Use the total number of points to randomly select number of centroids from the image
    n = len(x)
    centroids_init =  x[np.random.choice(n, num_clusters, replace=False), :]
    # *** END CODE HERE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START CODE HERE ***

    #Convert image to 2-d arr
    w,h,d = image.shape
    x = image.reshape((w * h, d))  
    
    #Initialize new_centroids
    new_centroids = np.empty(centroids.shape)
    print("This is the shape of centroids: ", centroids.shape)
    print("This is centroid: ", centroids)
    print("\n\n")
    print("This is centroid_0: ", centroids[0])
    temp = np.repeat(centroids[0, np.newaxis], x.shape[0], axis=0)
    print("This is temp: ", temp)
    print("This is temp shape: ", temp.shape)
    temp_2 = np.repeat(centroids[1, np.newaxis], x.shape[0], axis=0)
    print("This is temp_2: ", temp_2)
    print("This is temp_2 shape: ", temp_2.shape)
    result = np.append(temp, temp_2, axis=1)
    print("This is result: ", result)
    print("This is result shape: ", result.shape)
    
    
    print("This is x shape: ", x.shape)
    print("This is x: ", x)
    
    
    
    
    
#     centroids = centroids.reshape(16,3,1)
#     print("\n\n")
#     print("After reshape")
#     print("This is the shape of centroids: ", centroids.shape)
#     print("This is centroid: ", centroids)

    
    
    #Set max_iter
    max_iter = 1
    
    print("This is shape of x: ", x.shape)
    print("This is original x: ", x)
    x_new = np.repeat(x[:, :, np.newaxis], 16, axis=2)
    
    print("This is x new: ", x_new)
    print("This is shape of x_new: ", x_new.shape)
    print("x new 0: ", x_new[:, :, 0])
    print("x new 1: ", x_new[:, :, 1])
    print("x new 2: ", x_new[:, :, 2])
    print("This is x new 0 shape: ", x_new[:,:,0].shape)
    
    print(25*"==")
    
    i=0
    for each_centroid in centroids:
        print("This is each centroid: ", each_centroid)
        print("This is each centroid shape: ", each_centroid.shape)
        each_centroid.reshape(1, 3)
        print("This is each centroid after reshape: ", each_centroid)
        print("This is each centroid shape after reshape: ", each_centroid.shape)

        centroid_arr = np.repeat(each_centroid[:, np.newaxis], x.shape[0], axis=0)
        print("This is centroid_arr: ", centroid_arr)
        print("This is centroid_arr shape: ", centroid_arr.shape)
        print("\n\n")
    
    
    new_centroids_large = np.repeat(centroids[:,:,np.newaxis],x.shape[0], axis=2)
    print("This is new_centroids: ", new_centroids_large)
    print("This is new_centroids shape: ", new_centroids_large.shape)
    
    
    
    
    #Loop over max_iter
    for iter in range(max_iter):
        
        #Initialize an empty_arr to store mapping of each point to the closest centroid
        centroid_map = []
        
        #Function to map a centroid to a given input_val
        def map_centroid(input_val):
            
            print("This is input_val inside map_centroid: ", input_val.shape)
            
            def calc_distance(each_centroid):
               
                # finding sum of squares
                sum_sq = np.sum(np.square(input_val - each_centroid))

                # Doing squareroot and
                # printing Euclidean distance
                euclid_distance = np.sqrt(sum_sq)
                return euclid_distance
            
            mapped_centroid_index = np.argmin(calc_distance(centroids))
            
            return mapped_centroid_index
        
        centroid_map = map_centroid(x)
        
        print("This is centroid_map: ", centroid_map)
        print("This is centroid_map shape: ", centroid_map.shape)
            
        
        
        
        
    
        #Loop over all points - x and create their centroid_map
        #For each point, calculate the distance between the point and each centroid_map
        #Find the index of the centroid with min distance
        #Update in centroid_map arr
        for val_index in range(len(x)):
            dist_array = []
            for each_centroid in centroids:
                # finding sum of squares
                sum_sq = np.sum(np.square(x[val_index] - each_centroid))

                # Doing squareroot and
                # printing Euclidean distance
                euclid_distance = np.sqrt(sum_sq)
                dist_array.append(euclid_distance)

            minpos = dist_array.index(min(dist_array))
            centroid_map.append(minpos)


        #Mapping of each point to a given centroid
        centroid_map = np.array(centroid_map)



        #Based on centroid_map take the average values for a given centroid and update its value
        for index in range(len(centroids)):
            #Create a mask using centroid_map
            mask = (centroid_map == index)
            #Find out points for a given centroid
            extract_from_x = x[mask] # or,  a[a%3==0]
            #Take the average and update new_centroids
            new_centroids[index,:] = np.mean(extract_from_x, axis=0)
        
        print("Finished iteration: ", iter)    
    
    # *** END CODE HERE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START CODE HERE ***
#     print("Inside update_image")
#     print("This is image shape inside update_image: ", image.shape)
    w,h,d = image.shape
    x = image.reshape((w * h, d)) 
    
    #Function to find centroid which is closest to a given point
    def find_min_dist(point, centroids):
        dist_array = []
        for each_centroid in centroids:
            # finding sum of squares
            sum_sq = np.sum(np.square(point - each_centroid))

            # Doing squareroot and
            # printing Euclidean distance
            euclid_distance = np.sqrt(sum_sq)
            dist_array.append(euclid_distance)

            minpos = dist_array.index(min(dist_array))
            
        return centroids[minpos]
    
    
    #For each point, find the closest centroid
    #Update the point with the value of the closest centroid
    for each_point in range(len(x)):
        x[each_point] = find_min_dist(x[each_point], centroids)
    
    image = x.reshape(image.shape)    
    # *** END CODE HERE ***

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
