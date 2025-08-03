import numpy as np

from astropy.convolution import convolve

from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog, deblend_sources

from scipy.ndimage import rotate

def prepare_rotated(image, subtract=False, rotate_ok=True):

    # Estimate the background:
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    if subtract:
        data -= bkg.background  # subtract the background

    # Smooth the image:
    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    image_c = convolve(image, kernel)

    # Detect objects in the image
    threshold = 1.5 * bkg.background_rms
    segment_map = detect_sources(image_c, threshold, npixels=10)

    #segment_map = deblend_sources(image_c, segment_map_p,
    #                              npixels=10, nlevels=4, contrast=0.001,
    #                              progress_bar=False)

    # Catalogize the identified objects:
    cat = SourceCatalog(image, segment_map, convolved_data=image_c)
    tab = cat.to_table()


    # Find the target galaxy in the image:
    label_gal = np.argmax(tab['area']) + 1 # We assume that the targest galaxy is the biggest source in the image   
    ys, xs = np.shape(image)
    ys = ys/2.
    xs = xs/2.
    
    xmin = list(tab['bbox_xmin'])
    xmax = list(tab['bbox_xmax'])
    ymin = list(tab['bbox_ymin'])
    ymax = list(tab['bbox_ymax'])  
    
    # In fact, some sources in the image can be larger than the target galaxy, so we select the source which intersects the center:
    for ii in range(len(xmin)):
        if xs > xmin[ii] and xs < xmax[ii] and ys > ymin[ii] and ys < ymax[ii]:
            label_gal = ii + 1
    
    
    

    # Create the mask: mask out all objects except for the target galaxy:
    mask = np.where((segment_map.data > 0) * (segment_map.data != label_gal), 1, 0)

    # Get some geometric parameters of the galaxy:
    gal_params = cat.get_labels(label_gal).to_table()



    theta = gal_params['orientation'].value[0] # the position angle
    print('\t\tThe position angle is:', theta)

    # Calculate the dimensions of the target galaxy:
    xmin = gal_params['bbox_xmin'].value[0]
    xmax = gal_params['bbox_xmax'].value[0]
    ymin = gal_params['bbox_ymin'].value[0]
    ymax = gal_params['bbox_ymax'].value[0]

    sma = (xmax-xmin) / 2.
    smb = (ymax-ymin) / 2.

    if rotate_ok:
        # Rotate the galaxy image:
        image_r = rotate(image, theta, reshape = False)

        # Rotate the maske as well:
        mask_r = np.where(rotate(mask, theta, reshape = False) > 0.5, 1, 0)

        image_c = convolve(image_r, kernel)
        segment_map = detect_sources(image_c, threshold, npixels=10)

        # Catalogize the identified objects:
        cat = SourceCatalog(image_r, segment_map, convolved_data=image_c)
        tab = cat.to_table()

        # Find the target galaxy in the image:
        label_gal = np.argmax(tab['area']) + 1

        # Get some geometric parameters of the galaxy:
        gal_params = cat.get_labels(label_gal).to_table()



        theta = gal_params['orientation'].value[0] # the position angle
        print('The position angle is:', theta)

        # Calculate the dimensions of the target galaxy:
        xmin = gal_params['bbox_xmin'].value[0]
        xmax = gal_params['bbox_xmax'].value[0]
        ymin = gal_params['bbox_ymin'].value[0]
        ymax = gal_params['bbox_ymax'].value[0]

        sma = (xmax-xmin) / 2.
        smb = (ymax-ymin) / 2.
        print("Sma, Smb:", sma, smb)


        return (image_r, mask_r, theta, sma, smb)
    else:
        return (image, mask, theta, sma, smb)