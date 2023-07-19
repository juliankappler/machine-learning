#!/usr/bin/env python

import numpy as np
import copy 
import multiprocessing
import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg





class k_means():
    
    def __init__(self,K=2):
        #
        # set number of clusters
        self.K = K 
        #
        # initialize random number generator
        self.rng = np.random.default_rng()

    def load_data(self,data,verbose=False):
        ''' 
        Load data into the instance.

        Note that we create a class-internal copy of the data. This might be
        inconvenient for a very large datasets.
        '''
        
        # we obtain the number of datapoints and the dimension of the data
        # by examining the input
        self.N_data = len(data)
        self.N_dim = len(data[0])

        #
        if self.N_data < self.K:
            #
            error_msg = ("Input data must contain at least K = {K} ",
                         "datapoints, but only contains {N_data} datapoints.")
            raise RuntimeError(error_msg.format(K=self.K,N_data=self.N_data))

        # we store the data in a class-internal array
        self.data = np.zeros([self.N_data,self.N_dim],dtype=float)
        for i,e in enumerate(data):
            self.data[i] = copy.deepcopy(e)
        #
        if verbose:
            print("Loaded data, shape = {0}".format(np.shape(self.data)))
    
    def initialize_cluster_centroids(self):
        '''
        Initialize K cluster centroids

        To obtain the initial positions of the centroids, we randomly select
        K points from the input data (we sample without replacement).
        '''
        #
        self.centroids = np.zeros([self.K,self.N_dim])
        #
        # Note that we want unique centroids. We therefore cannot just 
        # randomly pick self.K items from self.data, as especially
        # for data with discrete values (such as images) it might be that
        # although i != j we have self.data[i] == self.data[j].
        # We therefore add the centroids one by one and make sure that any
        # new centroid is not already present in our list of centroids
        i = 0
        while (i < self.K):
            index = self.rng.choice(self.N_data,size=1)
            centroid_candidate = self.data[index]
            #
            if i == 0:
                self.centroids[i] = np.copy(centroid_candidate)
                i += 1
            else:
                diff = np.sum( (self.centroids[:i] - centroid_candidate)**2 ,
                              axis=-1)
                # diff[j] == 0 if and only if the current centroid candidate
                # is already present in self.centroids[:i]
                # We only want to add the current candidate if it does not
                # appear in the list of centroids we have already
                if not (diff == 0).any():
                    #
                    self.centroids[i] = np.copy(centroid_candidate)
                    i += 1
        #
        self.calculate_distances()
        self.assign_clusters()
        self.calculate_cost()

    def calculate_distances(self):
        '''
        Calculate all distances between the cluster centroids and the data

        This method creates an instance-internal array self.distances of 
        shape (K, N_data), such that 
        self.distances[k,i] = distance between centroid k and datapoint i

        "Distance" here refers to the square of the standard Euclidean norm
        '''
        #
        try:
            self.centroids
        except AttributeError:
            self.initialize_cluster_centroids()
        #
        diffs = self.centroids[:,np.newaxis] - self.data
        #
        self.distances = np.sum( diffs**2 , axis=-1 )

    def assign_clusters(self):
        '''
        Assign a cluster to each datapoint
        '''
        #
        try:
            # if self.distances does not exist, then the distances have never
            # been calculated
            self.distances
        except AttributeError:
            self.calculate_distances()
        #
        self.cluster_indices = np.argmin(self.distances,axis=0)

    def calculate_cost(self):
        '''
        Calculate the cost function (sometimes called "distortion")
        '''
        self.cost = 0.
        for i, index in enumerate(self.cluster_indices):
            self.cost += self.distances[index,i]

    def remove_empty_clusters(self,verbose=False):
        '''
        Remove all cluster centroids that currently have no datapoint
        associated with them.

        Note that if a centroid is removed, this will decrease self.K
        '''
        #
        # create cluster centroids if they do not exist yet
        # (this is to avoid an error if remove_empty_clusters is called 
        #  before cluster centroids have been initialized)
        try:
            self.cluster_indices
        except AttributeError:
            self.initialize_cluster_centroids()
        #
        #
        cluster_indices_in_data = np.unique(self.cluster_indices)
        #
        delete = [] # indices of the cluster centroids to be removed
        #
        for k in range(self.K):
            # for each cluster centroid k, check whether the number of 
            # datapoints associated with it are zero
            #if len( np.where( == k)[0] ) == 0:
            if k not in cluster_indices_in_data:
                # if so, append the current cluster to the list of clusters
                # that we want to remove
                delete.append(k)
        #
        # check if there are cluster centroids we want to remove
        if len(delete) > 0: 
            #
            if verbose:
                print("Removing cluster centroids {0}".format(delete))
                print('cluster_indices_in_data =',cluster_indices_in_data)
            #
            delete = np.array(delete)
            #
            # remove cluster centroids
            self.centroids = np.delete(arr=self.centroids,
                                    obj=delete,
                                    axis=0)
            #
            # if we remove a cluster centroid k, all indices i > k need to be 
            # decreased by 1. Since more than one cluster centroid can be 
            # removed at one step, for each cluster index we need to count 
            # how many cluster centroids with smaller index have been removed
            shifts = np.sum( (self.cluster_indices[:,np.newaxis] > delete) ,
                                    axis=-1)
            # shifts[i] = number of elements d in the array "delete" which
            #             fulfill cluster_indices[i] > d
            #           = how many cluster centroids that are being removed
            #             have an index smaller than cluster_indices[i] ?
            self.cluster_indices -= shifts
            #
            self.K -= len(delete)

    def update_cluster_centroids(self):
        #
        for k in range(self.K):
            self.centroids[k] = np.mean(self.data[np.where(self.cluster_indices==k)],axis=0)

    def step(self,return_centroids=False,verbose=False):
        #
        self.update_cluster_centroids()
        #
        self.calculate_distances()
        #
        self.assign_clusters()
        #
        self.remove_empty_clusters(verbose=verbose)
        #
        self.calculate_cost()
        #
        if return_centroids:
            return self.get_centroids()
        
    def run(self,
            data=None,
            N_steps=1000,
            N_runs=1,
            verbose=True):
        #
        if data is not None:
            #
            self.load_data(data=data)
        #
        minimal_cost = np.inf
        #
        K = self.K
        #
        for i in range(N_runs):
            #
            self.K = K
            self.initialize_cluster_centroids()
            #
            list_of_centroids = [copy.deepcopy(self.centroids)]
            list_of_costs = [self.get_cost()]
            for j in range(N_steps):
                if verbose:
                    print("Run {0}, step {1}".format(i+1,j+1),end='\r')
                #
                self.step(verbose=verbose)
                #
                list_of_centroids.append(copy.deepcopy(self.centroids))
                #
                list_of_costs.append(self.get_cost())
                #
                if list_of_costs[-1] == list_of_costs[-2]:
                    if verbose:
                        print("Run {0} converged after {1} steps".format(i+1,
                                                                         j+1))
                    break
            #
            if list_of_costs[-1] < minimal_cost:
                minimal_centroids = np.array(list_of_centroids)
                minimal_costs = np.array(list_of_costs)
                minimal_cost = minimal_costs[-1]
                minimal_cluster_indices = self.get_cluster_indices()
                if verbose:
                    print("Found new minimal cost of {0:3.5f}".format(minimal_cost))
        #
        self.K = K
        #
        return minimal_centroids, minimal_costs, minimal_cluster_indices

    def get_cost(self):
        #
        return self.cost

    def get_centroids(self):
        #
        try:
            self.centroids
        except AttributeError:
            self.initialize_cluster_centroids()
        
        return self.centroids
    
    def get_cluster_indices(self):
        return self.cluster_indices
    




class k_means_parallel(k_means):

    def __init__(self,K=2,N_cores=None):
        #
        # set number of clusters
        self.K = K 
        #
        # set number of CPUs to use
        if N_cores is None:
            self.N_cores = multiprocessing.cpu_count()
        else:
            self.N_cores = N_cores
        #

    def run_single(self,K,
                   data,
                   N_steps,
                   verbose=False):
        #
        k_means_instance = k_means(K=K)
        #
        minimal_centroids, minimal_costs, minimal_cluster_indices = \
            k_means_instance.run(data=data,
                                    N_steps=N_steps,
                                    N_runs=1,
                                    verbose=verbose)
        #
        output_dictionary = {'centroids':minimal_centroids,
                             'costs':minimal_costs,
                             'cluster_indices':minimal_cluster_indices}
        return output_dictionary
        

    def run(self,
            K=None,
            data=None,
            N_steps=1000,
            N_runs=10,
            N_cores=None,
            verbose=False):
        #
        if K is None:
            K = self.K
        #
        if data is not None:
            self.load_data(data=data)
        #
        if N_cores is None:
            N_cores = min(N_runs,self.N_cores)
        #
        list_of_results = joblib.Parallel(
                    n_jobs=N_cores)(
                                joblib.delayed(self.run_single)(
                                    K=K,
                                    data=data,
                                    N_steps=N_steps,
                                    verbose=verbose) for i in range(N_runs)
                                    )
        #
        minimal_cost = np.inf
        #
        for i,e in enumerate(list_of_results):
            #
            final_cost = e['costs'][-1]
            #
            if final_cost < minimal_cost:
                #
                minimal_cost = final_cost
                #
                minimal_centroids = e['centroids']
                minimal_costs = e['costs']
                minimal_cluster_indices = e['cluster_indices']
                if verbose:
                    print("Found new minimal cost of {0:3.5f}".format(minimal_cost))
        #
        return minimal_centroids, minimal_costs, minimal_cluster_indices
    


def plot_image(image,filename=None,title=None):
        #
        # get aspect ratio (used to ensure saved image has size similar to 
        # input image)
        x,y = np.shape(image)[:2]
        aspect_ratio = x/y
        figsize=5
        # 
        #
        fig, ax = plt.subplots(1,1,figsize=(figsize*aspect_ratio,figsize))
        fig.subplots_adjust(top = 1, bottom = 0, 
                            right = 1, left = 0, 
                            hspace = 0, wspace = 0)
        ax.imshow(image)
        ax.axis('off')
        if title is not None:
            ax.set_title(title,fontsize=aspect_ratio*15)
        plt.show()
        if filename is not None:
            fig.savefig(filename,bbox_inches='tight',dpi=int(x//figsize))
        plt.close(fig)

def decompress_image(image_compressed,
                         colors,
                         plot=True,
                         filename=None,
                         title =None,
                         **kwargs):
        #
        image = np.zeros((*np.shape(image_compressed),3),dtype=int)
        #
        #
        for i,e in enumerate(colors):
            mask = (image_compressed == i)
            image[mask] = e
        #
        if plot:
            plot_image(image=image,
                            filename=filename,
                            title=title)
        #
        return image
    


class image_compression():
    
    def __init__(self,K=2,
                        N_minimization=10,
                        N_steps=1000,
                        N_cores=None):
        #
        self.K = K
        self.N_minimization = N_minimization
        self.N_steps = N_steps
        #
        if N_cores is None:
            self.N_cores = multiprocessing.cpu_count()
        else:
            self.N_cores = N_cores
        
    def load_image_from_file(self,filename):
        #
        image = mpimg.imread(filename)
        self.import_image(image=image)

    def import_image(self,image):
        #
        self.image = np.copy(image)
    
    def compress_image(self,K=None,
                            image=None,
                            N_minimization=None,
                            N_steps=None,
                            N_cores=None):
        #
        if K is None:
            K = self.K
        if N_minimization is None:
            N_minimization = self.N_minimization
        if N_steps is None:
            N_steps = self.N_steps
        if N_cores is None:
            N_cores = min(self.N_cores,N_minimization)
        #
        if image is not None:
            pass
        else:
            try: 
                image = self.image;
            except NameError:
                error_msg = ("No image loaded. Please provide an image"
                             " via the function argument 'image'")
                raise RuntimeError(error_msg)
        #
        #
        image_flattened = image.reshape([np.prod(np.shape(image)[:-1]),3])
        #
        k_means_instance = k_means_parallel(K=K,N_cores=N_cores)
        #
        list_of_centroids, list_of_costs, cluster_indices = k_means_instance.run(
                                                        data=image_flattened,
                                                        N_steps=N_steps,
                                                        N_runs=N_minimization)
        #
        #
        colors = np.array(np.round(list_of_centroids[-1]),dtype=int)
        image_compressed = cluster_indices.reshape(np.shape(image)[:2])
        #
        output_dictionary = {'K':K,
                        'N_minimizations':N_minimization,
                        'N_steps':N_steps,
                        'colors':colors,
                        'image_compressed':image_compressed}
        #
        return output_dictionary
    
    
    def plot_image(self,image,filename=None,title=None):
        #
        return plot_image(image=image,
                          filename=filename,
                          title=title)

    def decompress_image(self,
                         image_compressed,
                         colors,
                         plot=True,
                         filename=None,
                         title =None,
                         **kwargs):
        #
        return decompress_image(
                         image_compressed,
                         colors,
                         plot,
                         filename,
                         title,
                         **kwargs)
