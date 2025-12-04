from IPython.display import display, Markdown
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd

def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display(Markdown(s))

def central(x):
    return np.mean(x), np.median(x), stats.mode(x).mode

def dispersion(x):
    y0 = np.std(x)
    y1 = np.min(x)
    y2 = np.max(x)
    y3 = y2 - y1
    y4 = np.percentile(x, 25)
    y5 = np.percentile(x, 75)
    y6 = y5 - y4
    return y0, y1, y2, y3, y4, y5, y6

def display_central_tendency_table(df, num=1):
    display_title('Central tendency summary statistics.', pref='Table', num=num)
    df_central = df.apply(lambda x: central(x), axis=0)
    round_dict = {col: 3 for col in df.columns}
    df_central = df_central.round(round_dict)
    df_central.index = ['mean','median','mode']
    display(df_central)

def display_dispersion_table(df, num=1):
    display_title('Dispersion summary statistics.', pref='Table', num=num)
    df_dispersion = df.apply(lambda x: dispersion(x), axis=0)
    round_dict = {col: 3 for col in df.columns}
    df_dispersion = df_dispersion.round(round_dict)
    df_dispersion.index = ['st.dev.','min','max','range','25th','75th','IQR']
    display(df_dispersion)

def plot_scatter_with_regression(df):
    y  = df['price']
    age  = df['age']
    frst = df['from station']
    nmcv = df['number of convenience stores']
    fig,axs = plt.subplots( 1, 3, figsize=(10,3), tight_layout=True )
    axs[0].scatter( age, y, alpha=0.5, color='b' )
    axs[1].scatter( frst, y, alpha=0.5, color='r' )
    axs[2].scatter( nmcv, y, alpha=0.5, color='g' )
    
    xlabels = 'Age', 'Distance from Station', 'Number of Convenience Stores' 
    [ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
    axs[0].set_ylabel('Price')
    [ax.set_yticklabels([])  for ax in axs[1:]]
    plt.show()

def plot_basic_scatter(df):
    y  = df['price']
    age  = df['age']
    frst = df['from station']
    nmcv = df['number of convenience stores']
    
    frst1 = np.around(frst/1000, 1)  # transformed density value
    
    fig,axs = plt.subplots( 1, 3, figsize=(10,3), tight_layout=True )
    axs[0].scatter( age, y, alpha=0.5, color='b' )
    axs[1].scatter( frst1, y, alpha=0.5, color='r' )
    axs[2].scatter( nmcv, y, alpha=0.5, color='g' )
    
    xlabels = 'Age', 'Distance from Station (1e-3)', 'Number of Convenience Stores' 
    [ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
    axs[1].set_xticks([0,1,2,3,4,5,6,7])
    axs[0].set_ylabel('Price')
    [ax.set_yticklabels([])  for ax in axs[1:]]
    plt.show()


def corrcoeff(x, y):
    r = np.corrcoef(x, y)[0,1]
    return r

def plot_regression_line(ax, x, y, **kwargs):
    a,b = np.polyfit(x, y, deg=1)
    x0,x1 = min(x), max(x)
    y0,y1 = a*x0+b, a*x1+b
    ax.plot([x0,x1],[y0,y1], **kwargs)


def plot_age_split_scatter(df):
    y   = df['price']
    age = df['age']
    i_low  = age <= 25
    i_high = age > 25
    
    fig, axs = plt.subplots(1, 2, figsize=(8,3), tight_layout=True)
    for ax, i in zip(axs, [i_low, i_high]):
        ax.scatter(age[i], y[i], alpha=0.5, color='g')
        plot_regression_line(ax, age[i], y[i], color='k', ls='-', lw=2)
    axs[0].set_title('Low-age houses')
    axs[1].set_title('High-age houses')
    axs[0].set_ylabel('Price')
    [ax.set_xlabel('Age') for ax in axs]
    plt.show()


def plot_age_split_mean(df):
    y   = df['price']
    age = df['age']
    i_low  = age <= 25
    i_high = age > 25
    
    fig, axs = plt.subplots(1, 2, figsize=(8,3), tight_layout=True)
    for ax, i in zip(axs, [i_low, i_high]):
        ax.scatter(age[i], y[i], alpha=0.5, color='g')
        plot_regression_line(ax, age[i], y[i], color='k', ls='-', lw=2)
    
    for a in np.unique(age[i_low]):
        axs[0].plot(a, y[i_low][age[i_low] == a].mean(), 'ro')
    for a in np.unique(age[i_high]):
        axs[1].plot(a, y[i_high][age[i_high] == a].mean(), 'ro')
        
    axs[0].set_title('Low-age houses')
    axs[1].set_title('High-age houses')
    axs[0].set_ylabel('Price')
    [ax.set_xlabel('Age') for ax in axs]
    plt.show()


def plot_descriptive(df):
    y   = df['price']
    age = df['age']
    frst = df['from station']
    nmcv = df['number of convenience stores']
    frst1 = np.around(frst/1000, 1)

    fig,axs = plt.subplots( 2, 2, figsize=(8,6), tight_layout=True )
    ivs     = [age, frst1, nmcv]
    colors  = 'b', 'r', 'g'
    for ax,x,c in zip(axs.ravel(), ivs, colors):
        ax.scatter( x, y, alpha=0.5, color=c )
        plot_regression_line(ax, x, y, color='k', ls='-', lw=2)
        r   = corrcoeff(x, y)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))
    
    xlabels = 'Age', 'Distance from Station (1e-3)', 'Number of Convenience Stores' 
    [ax.set_xlabel(s) for ax,s in zip(axs.ravel(),xlabels)]
    axs[0,1].set_xticks([0,1,2,3,4,5,6,7])
    [ax.set_ylabel('Price') for ax in axs[:,0]]
    [ax.set_yticklabels([]) for ax in axs[:,1]]

    ax       = axs[1,1]
    i_low    = age <= 25
    i_high   = age > 25
    fcolors  = 'm', 'c'
    labels   = 'Low-Age', 'High-Age'
    ylocs    = 0.3, 0.7
    for i,c,s,yloc in zip([i_low, i_high], fcolors, labels, ylocs):
        ax.scatter( age[i], y[i], alpha=0.5, color=c, facecolor=c, label=s )
        plot_regression_line(ax, age[i], y[i], color=c, ls='-', lw=2)
        r   = corrcoeff(age[i], y[i])
        ax.text(0.7, yloc, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

    ax.legend()
    ax.set_xlabel('Age')

    panel_labels = 'a', 'b', 'c', 'd'
    [ax.text(0.02, 0.92, f'({s})', size=12, transform=ax.transAxes)  for ax,s in zip(axs.ravel(), panel_labels)]
    plt.show()
    
    display_title('Correlations amongst main variables.', pref='Figure', num=1)
    plt.show()