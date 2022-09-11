import numpy as np

import warnings
warnings.filterwarnings('ignore')
import os, csv
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2, imageio
from tqdm import tqdm
import pandas as pd
from skimage import morphology as morph
from skimage import filters
from skimage import io
from skimage.morphology import disk, square
from skimage.filters.rank import mean
from scipy.ndimage.interpolation import rotate
from scipy import ndimage
from tqdm import tqdm  
import pickle as pkl
from scipy.ndimage.interpolation import rotate
import xlrd # use version 1.2
from skimage.morphology import area_opening, convex_hull_image, convex_hull_object
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
    
def anomaly_detection(data_dict, contamination=1e-10):
    df = pd.DataFrame.from_dict(data_dict)

    model = IsolationForest(contamination=contamination) # 0-0.5 or 'auto'
    scl = StandardScaler()

    scl_lr = scl.fit_transform(np.reshape(df['LR'].values, (-1,1)))
    pred = model.fit_predict(scl_lr) # outlier is -1, inlier is 1
    aLR = df.loc[pred == -1, ['LR']]

    scl_ll = scl.fit_transform(np.reshape(df['LL'].values, (-1,1)))
    pred = model.predict(scl_ll)
    aLL = df.loc[pred == -1, ['LL']]

    scl_ur = scl.fit_transform(np.reshape(df['UR'].values, (-1,1)))
    pred = model.predict(scl_ur)
    aUR = df.loc[pred == -1, ['UR']]

    scl_ul = scl.fit_transform(np.reshape(df['UL'].values, (-1,1)))
    pred = model.predict(scl_ul)
    aUL = df.loc[pred == -1, ['UL']]
    
    fig, ax = plt.subplots(figsize=(16,4))
    ax.plot(df.index, df['LR'], color='m', label = 'LR')
    ax.plot(df.index, df['LL'], color='blue', label = 'LL')
    ax.plot(df.index, df['UR'], color='green', label = 'UR')
    ax.plot(df.index, df['UL'], color='red', label = 'UR')

    ax.scatter(aLR.index, aLR['LR'], color='red', linewidths=3, label = 'Anomaly')
    ax.scatter(aLL.index, aLL['LL'], color='red', linewidths=3)
    ax.scatter(aUR.index, aUR['UR'], color='red', linewidths=3)
    ax.scatter(aUL.index, aUL['UL'], color='red', linewidths=3)

    plt.xlim([0, len(df['LR'])])
    #plt.ylim([0, 0.13])
    plt.legend()
    plt.show();
    
    
    

def plot_pressures(press_dict, 
                   shaded_areas=None, 
                   c=['r','b','g','m'], 
                   title=None, 
                   fig_size=(16,4), 
                   xlabel='Frame #', ylabel='Pressure ($mmHg$)'):
    plt.figure(figsize=fig_size)

    keys_lst = list(press_dict.keys())
    
    # get the first element of the dict
    x = range(len(press_dict[keys_lst[0]]))

    for i,k in enumerate(keys_lst):
        val = press_dict[k]
        if k == '$\mu$' or k == 'mean':
            color = 'gray' 
            ls = '--'
            lw = 1
        else:
            color = c[i]
            ls = '-'
            lw = 3
        plt.plot(x, val, c=color, ls=ls, lw=lw, label=k)

    if (shaded_areas is not None):
        keys_lst = list(shaded_areas.keys())
        for i,k in enumerate(keys_lst):
            vi = shaded_areas[k][0]
            vf = shaded_areas[k][1]
            
            plt.axvspan(vi, vf, color='gray', alpha=0.2, lw=0)
            
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
   
    plt.xticks(np.arange(0,len(x),10))
    plt.title(title);
    plt.legend()
    plt.grid(ls='--')
    
    plt.xlim([0, len(x)-1]);
    
    return


def plot_distributions(press_dict, c=['r','b','g','m'], title=None, fig_size=(16,4), xlabel='Pressure ($mmHg$)', kde=False):
    x_min, x_max = np.inf, 0
    
    plt.figure(figsize=fig_size)

    keys_lst = list(press_dict.keys())
    
    # get the first element of the dict
    x = range(len(press_dict[keys_lst[0]]))

    for i,k in enumerate(keys_lst):
        val = press_dict[k]
        # no useful to plot the mean
        if k == '$\mu$' or k == 'mean':
            continue
        color = c[i]
        
        sns.histplot(val, bins=100, color=color, edgecolor=None, label=k, alpha=.2, 
                     kde=kde, line_kws={'lw':3}, kde_kws={'cut':10, 'bw_adjust':.6});
        #sns.kdeplot(val, color=color, label=k, lw=3, common_norm=False);

        _, bin_edges = np.histogram(val, bins=100)
        x_min = np.min(bin_edges) if np.min(bin_edges) < x_min else x_min
        x_max = np.max(bin_edges) if np.max(bin_edges) > x_max else x_max
    
    #print(x_min, x_max)
    
    plt.xlim([x_min, x_max])
    plt.xlabel(xlabel)
    plt.grid(ls='--')
    plt.legend();

def plot_kdes(press_dict, c=['r','b','g','m'], title=None, fig_size=(16,4), xlabel='Pressure ($mmHg$)'):
    x_min, x_max = np.inf, 0
    
    plt.figure(figsize=fig_size)

    keys_lst = list(press_dict.keys())
    
    # get the first element of the dict
    x = range(len(press_dict[keys_lst[0]]))

    for i,k in enumerate(keys_lst):
        val = press_dict[k]
        # no useful to plot the mean
        if k == '$\mu$' or k == 'mean':
            continue
        color = c[i]
        
        sns.kdeplot(val, color=color, label=k);
    
        _, bin_edges = np.histogram(val, bins=100)
        x_min = np.min(bin_edges) if np.min(bin_edges) < x_min else x_min
        x_max = np.max(bin_edges) if np.max(bin_edges) > x_max else x_max
    
    #print(x_min, x_max)
    
    plt.xlim([x_min, x_max])
    plt.xlabel(xlabel)
    plt.grid(ls='--')
    plt.legend();


# c, I removed ,'g','m'
def plot_ecdf(press_dict, c=['r','b','g','m'], title=None, fig_size=(16,4), xlabel='Pressure ($mmHg$)'):
    x_min, x_max = np.inf, 0
    
    # https://www.geeksforgeeks.org/how-to-make-ecdf-plot-with-seaborn-in-python/
    df = pd.DataFrame(data=press_dict)

    plt.figure(figsize=(16,4))
    sns.ecdfplot(data=df, palette=c, lw=3)
    plt.grid(ls='--')
    plt.xlabel(xlabel);
    
    lst_values = list(press_dict.values())
    _, bin_edges = np.histogram(lst_values, bins=100)
    
    x_min = np.min(bin_edges)
    x_max = np.max(bin_edges)
    
    #print(x_min, x_max)
    
    plt.xlim([x_min, x_max])


def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')



def rolling_window(arr, n=2):
    # need two fillna to fill the gaps at the beginning and maybe at the end
    arr_final = pd.Series(arr).rolling(n).mean().fillna(method='bfill').fillna(method='ffill')
    
    return arr_final

def binarization(img, method='convex_hull'):
    #hull1 = convex_hull_image(img_f > 0, tolerance=10)
    if (method == 'convex_hull'):
        imf = convex_hull_object(img > 0, connectivity=2)
    
    return imf

def best_threshold(img):
    h, bins = np.histogram(img.ravel(), bins=255)
    t = filters.threshold_otsu(hist=h)
    
    return t


def frames_medaxis(data, dataset='PMat', t=2, axis=0, roll_window=10):
    frames_pts = {}
    
    # shape[0] to get the number of frames
    for fi in tqdm(range(data.shape[0])):
        #print(fi)
        if (dataset == 'PMat'):
            img_fi = data[fi,:].reshape((64,32))
        elif (dataset == 'XSensor'):
            img_fi = rotate(data[fi,:,:], 180, reshape=True)
        else:
            raise Exception("[dataset] not defined")
            
        _,img_bin = pre_processing(img_fi, t=t)    
         
        pts = find_medaxis(img_bin, axis=axis)
        # FIXIT: Just to ignore the rolling window in case of cutting the image on the horizontal.
        # the rolling window output will be wrong coordinates
        if (axis == 1):
            roll_window = 0
        if (roll_window > 0):
            #print(fi, pts)
            # FIXIT: unpacking
            pts_tmp = [int(p[1]) for p in pts]
            pts_roll = rolling_window(pts_tmp, n=roll_window)
            pts = [[k,v] for k,v in enumerate(pts_roll)]
            
        frames_pts[fi] = pts
        
    return frames_pts


def left_right_analysis(data, frames_pts, dataset='PMat'):
    left_means = []
    right_means = []

    for fi in tqdm(range(data.shape[0])):
        pts = frames_pts[fi]
        
        if (dataset == 'PMat'):
            img = data[fi,:].reshape((64,32))
        elif (dataset == 'XSensor'):
            img = data[fi,:,:]
        else:
            raise Exception("[dataset] not defined")

        left_prs = []
        right_prs = []

        for p in pts:
            #print(p)
            left_prs.append(img[p[0],0:int(p[1])].tolist())
            right_prs.append(img[p[0],int(p[1]):].tolist())

        left_flat = [item for sublist in left_prs for item in sublist]
        right_flat = [item for sublist in right_prs for item in sublist]

        left_means.append(np.mean(left_flat))
        right_means.append(np.mean(right_flat))

    return left_means, right_means


def relative_movement(data, measures, dataset='PMat'):
    '''
    Calculate the relative movement by dividing the left by the right side
    
    measures: 
    '''
    #TODO: Include the parameter to decide between left/right (default) or right/left
    res = {}
    
    #left = np.concatenate((np.divide(measures['UL'], np.divide(measures['LL']))
    #right = np.concatenate((np.divide(measures['UR'], np.divide(measures['LR']))
    
    res['U'] = np.divide(measures['UL'], measures['UR'])
    res['L'] = np.divide(measures['LL'], measures['LR'])
    
    return res


def quadrant_analysis(data, pts, dataset='PMat', func=None):
    
    if not func:
        func = np.mean
    
    res = {}
    res['UL'], res['LL'] = [], []
    res['UR'], res['LR'] = [], []

    pts_vert = pts['vert']
    pts_horiz = pts['horiz']
    
    print('Quadrant analysis...')
    for fi in tqdm(range(data.shape[0])):
        # getting the points set according to the frame
        pts_v = pts_vert[fi]
        pts_h = pts_horiz[fi]
        
        if (dataset == 'PMat'):
            img = data[fi,:].reshape((64,32))
        elif (dataset == 'XSensor'):
            img = data[fi,:,:]
        else:
            raise Exception("[dataset] not defined")

        UL_prs, UR_prs = [], []        
        LL_prs, LR_prs = [], [] 

        for p in pts_v:
            # Upper part
            if (p[0] < pts_h[0][0]):
                UL_prs.append(img[p[0],0:int(p[1])].tolist())
                UR_prs.append(img[p[0],int(p[1]):].tolist())
            # Lower part
            else:
                LL_prs.append(img[p[0],0:int(p[1])].tolist())
                LR_prs.append(img[p[0],int(p[1]):].tolist())                

        UL_flat = [item for sublist in UL_prs for item in sublist]
        UR_flat = [item for sublist in UR_prs for item in sublist]
        LL_flat = [item for sublist in LL_prs for item in sublist]
        LR_flat = [item for sublist in LR_prs for item in sublist]
        
        res['UL'].append(func(UL_flat))
        res['UR'].append(func(UR_flat))
        res['LL'].append(func(LL_flat))
        res['LR'].append(func(LR_flat))

    return res


def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pkl.load(f)
    print('--> Data loaded!')
    
    return data


def data_to_pkl(data, fn):
    with open(fn, "wb") as f:
        pkl.dump(data, f)
    print('--> File saved: {}'.format(fn))


def pre_processing(img, t: int = 2):
    # normalization
    img_f = (img / np.max(img)) * 255

    img_f = filters.median(img_f,
                           selem=disk(3), 
                           mode='constant', 
                           cval=0)
    img_bin = img_f > t
    img_f = img_f * img_bin
    #print(type(img_f))
    
    img_f = (img_f / np.max(img_f)) * 255
    #print(type(img_f))
    
    return img_f, img_bin


def get_frame(data, frame: int, dataset, raw=False, mask=False, t: int = 2):
    if (dataset == 'PMat'):
        img_fi = data[frame,:].reshape((64,32))
    elif(dataset == 'XSensor'):
        img_fi = rotate(data[frame,:,:], 180, reshape=True)
        #print(img_fi.shape, type(img_fi))
        #return img_fi
    
    if not raw:
        img_fi,img_bin = pre_processing(img_fi, t)
    #print(type(img_fi))#(img_fi.shape)
    
    if (mask and not raw):
        return img_fi, img_bin
    else:
        return img_fi


def plot_img_marks(img, img_bin=None, dots_v: list=None, dots_h: list=None):
    plt.figure(figsize=(10,8))
    
    n = 1
    if (type(img_bin) is 'numpy.ndarray'):
        n = 2
        
    plt.subplot(1,n,1)
    plt.imshow(img)

    if (dots_v != None and dots_h != None):
        for p in dots_h:
            plt.plot(p[1], p[0], marker='o', color="m")
        for p in dots_v:
            plt.plot(p[1], p[0], marker='o', color="m")

    if (type(img_bin) is 'numpy.ndarray'):
        plt.subplot(1,n,2)
        plt.imshow(img_bin);
        
    plt.axis('off')


def find_medaxis(img, axis=0):
    '''
    return: list, [ [row,column], ...]
    '''
    r,c = img.shape

    dots_lst = []
    
    # vertical --> rows direction
    if (axis == 0):
        for ri in range(r):
            row = img[ri,:]
            rrr =  np.where(row > 0)[0]
            #print(len(rrr), np.shape(rrr), rrr, row)
            if (len(rrr) == 0):
                dots_lst.append([ri, np.ceil(len(row) // 2)])
                continue

            rrr = np.arange(rrr[0], rrr[-1]+1)
            #print(rrr)
            dots_lst.append([ri, int(np.take(rrr, np.ceil(len(rrr) // 2)))])
            #print(ri, ':', np.take(rrr, np.ceil(len(rrr) // 2)))
    
    # horizontal --> columns direction
    elif (axis == 1):
                
        for ci in range(c):
            col = img[:,ci]
            #rrr =  np.where(col > 0)[0]
            #if (len(rrr) == 0):
            #    dots_lst.append([np.ceil(len(col) // 2), ci])
            #    continue
            #rrr = np.arange(rrr[0], rrr[-1]+1)
            #print(rrr)
            dots_lst.append([int(2 * r / 5), int(ci)])
            
    else:
        raise Exception("wrong [axis] value")
        
    return dots_lst

def clean_csv(in_fn, out_fn):
    #in_fn = 'P12PH_t5_PS0008R4S0039_20210301_214352_PSMLAB_csv.csv'
    #out_fn = 'P12PH_t5_PS0008R4S0039_20210301_214352_PSMLAB_csv2.csv'

    with open(in_fn) as in_file:
        with open(out_fn, 'w', newline='\n') as out_file:
            writer = csv.writer(out_file)
            for row in tqdm(csv.reader(in_file)):
                #print(row)
                if len(row) > 2 and row[0].replace('.', '', 1).isdigit():
                    writer.writerow(row)
                # if row and row[0] not in ['', 'COP Column:', 'COP Row:', 'Date:',
                #                     'Frame:', 'Peak Pressure (N/cm2):',
                #                     'Rows:', 'Time:', 'Sensor:', 'Columns:',
                #                     'Sensel Width (cm):', 'Sensel Height (cm):',
                #                     'Average Pressure (N/cm2):', 
                #                     'Contact Area (cm^2):', 'Sensels:',
                #                     'Units:', 'Threshold:', '\r\n']:
                #     writer.writerow(row)


def read_excel_data(FileName):
    """
    Open excel file an populates a numpy array.

    Excel file must contain the pressure data in the second sheet. The pressure
    data should be of shape 48 rows by 118 columns.

    Parameters
    ----------
    FileName : str
        Pathname of the file that needs to be read.

    Returns
    -------
    num_arr : ndarray
        Pressure data array in shape = (samples, cols, rows).

    """
    #from openpyxl import load_workbook
    # wb = load_workbook(filename = 'empty_book.xlsx')
    # sheet_ranges = wb['range names']
    
    wb = xlrd.open_workbook(FileName)
    
    # 'processed' sheet
    sheet = wb.sheet_by_index(2)
    
    Data_Frame_Number = int(sheet.nrows / 48)
    num_arr = np.zeros((48, 118, Data_Frame_Number))

    frame = 0
    row = 0
    for i in range(sheet.nrows):
        if i < 48:
            row = i
        else:
            row = i % 48
        if i != 0 and (i % 48 == 0):
            frame += 1
        for col in range(118):
            #print(i, col)
            num_arr[row][col][frame] = sheet.cell_value(i, col)

    wb.release_resources()
    del wb
    num_arr = np.swapaxes(num_arr, 0, 2)
    return num_arr


def read_excel_data_pandas(FileName):
    # 'processed' sheet
    sheet = pd.read_excel(FileName, sheet_name=2)
    print('sheet:',sheet.shape)
    nrows = sheet.shape[0]
    
    Data_Frame_Number = int(nrows / 48) + 1 # FIXIT: check why I need the +1
    num_arr = np.zeros((48, 118, Data_Frame_Number))
    print(num_arr.shape)

    frame = 0
    row = 0
    for i in tqdm(range(nrows)):
        if i < 48:
            row = i
        else:
            row = i % 48
        if i != 0 and (i % 48 == 0):
            frame += 1
        for col in range(118):
            #print(i, col)
            val = sheet.iloc[i, col]
            num_arr[row][col][frame] = val

    num_arr = np.swapaxes(num_arr, 0, 2)
    return num_arr


def read_csv_data(FileName):    
    # 'processed' sheet
    sheet = pd.read_csv(FileName, header=None)
    print('sheet:',sheet.shape)
    nrows = sheet.shape[0]
    
    Data_Frame_Number = int(nrows / 48) + 1 # FIXIT: check why I need the +1
    num_arr = np.zeros((48, 118, Data_Frame_Number))
    print(num_arr.shape)

    frame = 0
    row = 0
    for i in tqdm(range(nrows)):
        if i < 48:
            row = i
        else:
            row = i % 48
        if i != 0 and (i % 48 == 0):
            frame += 1
        for col in range(118):
            #print(i, col)
            val = sheet.iloc[i, col]
            num_arr[row][col][frame] = val

    num_arr = np.swapaxes(num_arr, 0, 2)
    return num_arr
