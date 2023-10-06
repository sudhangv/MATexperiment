
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import lmfit
from lmfit.lineshapes import gaussian2d, lorentzian
from PIL import Image


# image_path = r"C:\Users\svars\OneDrive\Desktop\UBC Lab\CATExperiment\CATMeasurements\VeffMeasurements\tests\0p4_82_exp100.jpg"  # Replace with the actual image path
# image = Image.open(image_path)

# image_array = np.array(image)
# roi = image_array[:, :]

# plt.imshow(roi)
def fit_gaussian2d(image_data, title_og='Original Image', title_fit='Fitted Image'):
    x = np.arange(image_data.shape[0])
    y = np.arange(image_data.shape[1])
    X,Y = np.meshgrid(x,y)
    
    # def rot_gaussian2D(x,y, amplitude=1, centerx=0, centery=0,
    #                 sigmax=1, sigmay=1):
    #     rotation=0
    #     xp = (x - centerx)*np.cos(rotation) - (y - centery)*np.sin(rotation)
    #     yp = (x - centerx)*np.sin(rotation) + (y - centery)*np.cos(rotation)
    #     return gaussian2d(xp, yp, amplitude=amplitude, centerx=centerx, centery=centery, sigmax=sigmax, sigmay=sigmay)
    
    # model = lmfit.Model(rot_gaussian2D, independent_vars=['x', 'y'])
    # params = model.make_params(amplitude=15723, centerx=X.ravel()[np.argmax(image_data.ravel())], \
    #     centery=Y.ravel()[np.argmax(image_data.ravel())], sigmax=5, sigmay=5)
    model = lmfit.models.Gaussian2dModel()
    params = model.guess(image_data.ravel(), X.ravel(), Y.ravel())
    result = model.fit(image_data.ravel(), x=X.ravel(), y=Y.ravel(), params=params)
    lmfit.report_fit(result)
    
    plt.subplot(121)
    plt.imshow(image_data)
    plt.title(title_og)
    plt.subplot(122)
    plt.imshow(result.best_fit.reshape(len(y), len(x)))
    plt.title(title_fit)
    
    return result.best_values


# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import griddata
# import os
# from os.path import join as join

# import lmfit
# from lmfit.lineshapes import gaussian2d, lorentzian
# from PIL import Image

# # Load the image
# run_folder = r"C:\Users\svars\OneDrive\Desktop\UBC Lab\CATExperiment\CATMeasurements\VeffMeasurements\Sept26Dispensation"

# j=0
# for i, img in enumerate(os.listdir(run_folder)[:]):
#     if '_83_' in img:
#         j = j +1
#         image_path = join(run_folder, img)
#         image = Image.open(image_path)

#     # Convert the image to a numpy array

#         image_array = np.array(image, dtype=float)
#         plot_data = image_array[:63, 70]
#         plot_data = plot_data- np.mean(plot_data[0:10])
        
#         y = plot_data
#         x = np.arange(len(plot_data))
#         model = lmfit.models.GaussianModel()
#         params = model.guess(y, x=x)
#         result = model.fit(y, x=x, params=params)
#         result.fit_report()

#         plt.plot(x, y, 'o', color=f'C{j}', markersize=3 )
#         plt.plot(x, result.best_fit, '-', label=fr"$\sigma$ = {result.params['sigma'].value:.2f}, name  = {img}", color=f'C{j}')
        
# plt.legend(bbox_to_anchor = [1.5, 1.5])