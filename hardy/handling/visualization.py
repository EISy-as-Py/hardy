import numpy as np
import matplotlib.pyplot as plt


#######################################################################
# Normalization Functions
def normalize(impedance_array):
    '''Function that returns a normalized impedance_array

    The function takes the maximum value of an array and divides each entry of
    the array by it. Additionally, if the minimum of the array is negative, it
    shifts it to zero, so that the resulting normalized array will have a range
    zero to one.

    Parameters
    ----------
    impedance_array : array-like
                      the array to be normalized.

    Returns
    -------
    normalized_impedance_array :  array-like
                                  the normalized array. All entries in this
                                  array should be values in the range
                                  zero to one.
    '''
    temp_array = impedance_array.copy()
    if np.amin(temp_array, axis=0) < 0:
        temp_array += abs(np.amin(temp_array, axis=0))
    if np.amax(temp_array, axis=0) > 1:
        normalized_impedance_array = temp_array/np.amax(temp_array, axis=0)
    else:
        normalized_impedance_array = temp_array

    return normalized_impedance_array


def normalize_image(color_image_array):
    ''' Function that normalizes a color image array.
    The color image array will ahve dimensions (n,n, 3).
    The 'n' value will depends on how big your image is

    Parameters
    ----------

    color_image_array : array-like
                        a multidimentional array of shape (n,n,3)

    Returns
    -------
    normalized_image: array-like
                      the normalized image array. All entries in this
                      array should be values in the range zero to one.
                      The image shape should still be (n,n,3)
    '''
    # Determine the size of the square image
    n = np.shape(color_image_array)[0]

    normalized_image = np.empty((n, n, 3), dtype=np.float32)

    for i in range(np.shape(color_image_array)[2]):
        img = color_image_array[:, :, i]
        if np.count_nonzero(img) != 0:
            # normalizing data per channel
            normalized_image[:, :, i] = img / (np.amax(img, axis=0))
        else:
            # skip any channel with all zero to avoid 'NaN' as output
            normalized_image[:, :, i] = img

    return normalized_image


######################################################################
# Plotting Functions
def rgb_plot(red_array=None, green_array=None, blue_array=None,
             plot=True, save_image=None, filename=None,
             save_location=None):
    '''Returns a plot which represents the input data as a color gradient of
    one of the three color channels available: red, blue or green.

    This function represents the data as a color gradient in one of the three
    basic colors: red, blue or green. The color gradient is represented on the
    x-axis, leaving the y-axis as an arbitrary one. This means that the size or
    the scale of the y-axis do not have a numerical significance. The input
    arrays shoudld be of range zero to one. A minimum of one array should be
    provided. The final representation will be a square plot of the combined
    arrays.

    Parameters
    ----------
    red_array : array
                the data array to be plotted in the red channel.
    green_array : array
                  the data array to be plotted in the green channel.
    blue_array : array
                 the data array to be plotted in the blue channel.
    plot : bool
           if True, the color gradient representation of the data will be
           displayed
    filename : str
               The filename will be the same as the .csv containing the data
               used to create this plot.
    save_location : str
                    String containing the path of the forlder to use when
                    saving the data and the image.
    save_image : bool
                 Option to save the output of the simuation as a plot
                 in a .png file format.
                 The filename used for the file will be the same
                 as the raw data file created in this function.
    Returns
    -------
    rbg_plot :  matplotlib plot
                Plot representing the data as a color gradient on the x-axis
                in one of the three basic colors: red, blue or green
    '''
    arrays = {'red_array': red_array, 'blue_array': blue_array,
              'green_array': green_array}

    given = {k: v is not None for i, (k, v) in enumerate(arrays.items())}
    given_arrays = [(k, arrays[k]) for i, (k, v) in enumerate(given.items())
                    if v is True]
    n = []
    for i in range(len(given_arrays)):
        n.append(len(given_arrays[i][1]))
    assert len(given_arrays) != 0, 'no input array was given.'
    assert all(x == n[0] for x in n), 'the given arrays have different length.\
        Check that you are using the right inputs'

    not_given = [k for (k, v) in given.items() if v is False]
    for array in not_given:
        arrays[array] = np.zeros(n[0])

    # Normalize Data from 0 to 1 (aka RGB readable)
    red_array = normalize(arrays['red_array'])
    green_array = normalize(arrays['green_array'])
    blue_array = normalize(arrays['blue_array'])

    arbitrary_axis = np.linspace(0, 1, n[0])

    r_big, a = np.meshgrid(red_array, arbitrary_axis)
    g_big, a = np.meshgrid(green_array, arbitrary_axis)
    b_big, a = np.meshgrid(blue_array, arbitrary_axis)

    rgb_plot = np.ndarray(shape=(n[0], n[0], 3))

    rgb_plot[:, :, 0] = r_big
    rgb_plot[:, :, 1] = g_big
    rgb_plot[:, :, 2] = b_big

    if plot:
        big, bax = plt.subplots(1, 1, figsize=[6, 6])
        bax.imshow(rgb_plot)
        bax.axis('off')

    if save_image:
        filename = str(save_location+filename)
        plt.savefig('{}.png'.format(filename), dpi=100, bbox_inches='tight')
        plt.close()

    return rgb_plot


def orthogonal_images_add(image_x, image_y, plot=True, save_image=None,
                          filename=None, save_location=None):
    """
    Takes two images and combines them by rotating one of
    them 90 degrees and adds the two up. The resulting array
    is then normalized by channel.
    Takes in two images of shape=(ARBITRARY, Data-axis, 3)

    Parameters
    ----------
    image_x : array-like
              A multidimentional array of shape (n,n,3) with
              entries in range zero to one
    image_y : array-like
              A multidimentional array of shape (n,n,3) with
              entries in range zero to one
    plot : bool
           if True, the color gradient representation of the data will be
           displayed
    filename : str
               The filename will be the same as the .csv containing the data
               used to create this plot.
    save_location : str
                    String containing the path of the forlder to use when
                    saving the data and the image.
    save_image : bool
                 Option to save the output of the simuation as a plot
                 in a .png file format.
                 The filename used for the file will be the same
                 as the raw data file created in this function.
    Returns
    -------
    combined_image :  matplotlib plot
                      Plot representing the data as a color gradient on the
                      x-axis and on the y-axis in one of the three basic
                      colors: red, blue or green

    """
    image_flip = np.ndarray(shape=image_y.shape)
    for channel in range(3):
        image_flip[:, :, channel] = image_y[:, :, channel].transpose()

    combined_image = normalize_image(image_x + image_flip)

    if plot:
        fig, ax = plt.subplots(figsize=[6, 6])
        ax.imshow(combined_image)
        ax.axis('off')

    if save_image:
        filename = str(save_location+filename)
        plt.savefig('{}.png'.format(filename), dpi=100, bbox_inches='tight')
        plt.close()

    return combined_image


def orthogonal_images_mlt(image_x, image_y, plot=True, save_image=None,
                          filename=None, save_location=None):
    '''
    Takes two images and combines them by rotating one of
    them 90 degrees and multiplies them.
    Takes in two images of shape=(ARBITRARY, Data-axis, 3)

    NOTE: If one axis of a color (Red X) has data but the other (Red Y)
          has nothing, we should Replace the Zero-array with a Ones-Array!




    Parameters
    ----------
    image_x : array-like
              A multidimentional array of shape (n,n,3) with
              entries in range zero to one
    image_y : array-like
              A multidimentional array of shape (n,n,3) with
              entries in range zero to one
    plot : bool
           if True, the color gradient representation of the data will be
           displayed
    filename : str
               The filename will be the same as the .csv containing the data
               used to create this plot.
    save_location : str
                    String containing the path of the forlder to use when
                    saving the data and the image.
    save_image : bool
                 Option to save the output of the simuation as a plot
                 in a .png file format.
                 The filename used for the file will be the same
                 as the raw data file created in this function.
    Returns
    -------
    combined_image :  matplotlib plot
                      Plot representing the data as a color gradient on the
                      x-axis and on the y-axis in one of the three basic
                      colors: red, blue or green
    '''
    # Check and Fix/Prepare for Zero-Arrays
    for channel in range(3):
        if 0 != image_x[:, :, channel].all():
            if 0 == image_y[:, :, channel].all():
                # IF X has data, but Y does not, Make Y all 1's
                image_y[:, :, channel] = np.ones_like(image_y[:, :, channel])
            else:
                pass
        elif 0 != image_y[:, :, channel].all():
            if 0 == image_x[:, :, channel].all():
                # IF Y has data, but X does not, Make X all 1's
                image_x[:, :, channel] = np.ones_like(image_x[:, :, channel])
            else:
                pass
        else:
            pass

    image_flip = np.ndarray(shape=image_y.shape)
    for channel in range(3):
        image_flip[:, :, channel] = image_y[:, :, channel].transpose()

    combined_image = (image_x * image_flip)

    if plot:
        fig, ax = plt.subplots(figsize=[6, 6])
        ax.imshow(combined_image)
        ax.axis('off')

    if save_image:
        filename = str(save_location+filename)
        plt.savefig('{}.png'.format(filename), dpi=100, bbox_inches='tight')
        plt.close()
    return combined_image


# add new plotting functions here [ Maria !!]
