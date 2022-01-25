import scipy
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    dataarray = np.load(filename)
    dataarray = dataarray - np.mean(dataarray, axis=0)
    return dataarray


def get_covariance(dataset):
    res_array = np.empty(shape=(1024, 1024))  # makes an empty res array
    for x in dataset:
        xprime = x.reshape(1024, 1)
        xt = np.transpose(xprime)  # transposes x
        arr_temp = np.dot(xprime, xt)  # the dot product between x transposed
        res_array = np.add(res_array, arr_temp)  # matrix addition

    scalar = 1 / (len(dataset)-1)  ## the scaler
    res_array = res_array * scalar  # multiplies the scaler by the matrix
    return res_array


def get_eig(S, m):
    w,v= scipy.linalg.eigh(S) # gets the eigenvalues and vectors as a tuple
    w_notfliped = w
    w = np.flipud(w)  # reverses the order of the eigenvalues so it is in descending order
    v = v  # all the eigenvectors
    # instantiates all of the result matrixes
    egval_arr = np.zeros(shape=(m, m))
    egvector_arr = np.zeros(shape=(m, w.shape[0]))
    egvector_arr = np.transpose(egvector_arr)
    # adds all the eigenvalues
    for i in range(0, m):
        egval_arr[i][i] = w[i]
        index = np.where(w_notfliped == w[i])
        ind = int(index[0])
        # ads the eigenvectors
        for j in range(0, w.shape[0]):
            egvector_arr[j][i] = v[j][ind]
    return egval_arr, egvector_arr


def get_eig_perc(S, perc):
    w,v= scipy.linalg.eigh(S)
    w_notfliped = w    # gets the eigenvalues and vectors as a tuple
    w = np.flipud(w) # reverses the order of the eigenvalues so it is in descending order
    v = v # all the eigenvectors
    lis = list_invarice(w, perc) # calls a helper method to get a
    # list of all the eigenvalues that are greater then percs
    # instantiates all of the result matrixes
    egval_arr = np.zeros(shape=(len(lis), len(lis)))
    egvector_arr = np.zeros(shape=(len(lis), w.shape[0]))
    egvector_arr = np.transpose(egvector_arr)
    # adds all of the values into the empty matrixs
    for i in range(0, len(lis)):
        egval_arr[i][i] = lis[i]
        index = np.where(w_notfliped == w[i])
        ind = int(index[0])
        for j in range(0, w.shape[0]):
            egvector_arr[j][i] = v[j][ind]
    return egval_arr, egvector_arr


def project_image(img, U):
    res_array = np.zeros(shape=(1, 1024)) #result array
    for i in range(0, U.shape[1]):
        temparr = U[:, i] # gets coloum i
        Utrans = np.transpose(temparr) # transposes it
        aij = np.dot(Utrans, img) # get the Aij from the formula
        temp = np.dot(aij, temparr) # gets the second part
        res_array = np.add(res_array, temp) # adds it all together

    return res_array


def display_image(orig, proj):
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title('Original')
    axs[1].set_title('Projected')
    # reshapes and transposes the two
    orig = np.reshape(orig, (-1, 32))
    orig = np.transpose(orig)
    proj = np.reshape(proj, (-1, 32))
    proj = np.transpose(proj)
    # adds them
    one = axs[0].imshow(orig, aspect='equal')
    two = axs[1].imshow(proj, aspect='equal')
    # adds the color bars
    fig.colorbar(one, ax=axs[0])
    fig.colorbar(two, ax=axs[1])
    plt.show()
    return 0


def list_invarice(vals, varience):
    sum = 0
    for i in range(0, vals.shape[0]):
        temp = int(vals[i])
        sum += temp
    lis = []
    # adding the eigenvalues that are greater than variance
    for j in range(0, vals.shape[0]):
        if (vals[j] / sum >= varience):
            lis.append(vals[j])
    return lis


x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)

Lambda, U = get_eig(S, 2)
print(Lambda)
print(U)
projection = project_image(x[0], U)
display_image(x[0], projection)