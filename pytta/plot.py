""" 
plot:
------

    This plot module receives organized raw data (numpy arrays, lists and
    strings) and plot it through its functions. It is used by the function,
    signal, and analysis modules.

    Available functions:
    ---------------------

        >>> plot.time(dataSet, xLabel, yLabel, yLim, xLim, title, decimalSep)
        >>> plot.freq(dataSet, smooth, xLabel, yLabel, yLim, xLim, title,
                      decimalSep)
        >>> plot.bars(dataSet, xLabel, yLabel, yLim, title, decimalSep,
                      barWidth, errorStyle)

    For further information check the function especific documentation.
    
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale
import numpy as np
import scipy.signal as ss
import copy as cp

def time(dataSet, xLabel, yLabel, yLim, xLim, title, decimalSep):
    """Plot a signal in time domain.

    Parameters (default), (type):
    -----------------------------

        * dataSet (), (list):
            A list with dictionaries containing the information about each
            curve. Needed keys for time plot:

                - 'x', (), (ndarray): time axis;
                - 'y', (), (ndarray): amplitude axis;
                - 'label', (), (str): curve label.

                >>> dataSet = [{'x':x, 'y':y, 'label':'my beautiful curve}]
        
        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Amplitude'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * xLim (), (list):
            left and right limits

            >>> xLim = [0, 15]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

    Return:
    --------

        matplotlib.figure.Figure object.
    """
    if decimalSep == ',':
        locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
        plt.rcParams['axes.formatter.use_locale'] = True
    elif decimalSep =='.':
        locale.setlocale(locale.LC_NUMERIC, 'C')
        plt.rcParams['axes.formatter.use_locale'] = False
    else:
        raise ValueError("'decimalSep' must be the string '.' or ','.")

    if xLabel is None:
        xLabel = 'Time [s]'
    if yLabel is None:
        yLabel = 'Amplitude'
    if title is None:
        title = ''

    xLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)
    yLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.10, 0.21, 0.88, 0.72], polar=False,
                        projection='rectilinear', xscale='linear')
    ax.set_snap(True)
    ax.grid(color='gray', linestyle='-.', linewidth=0.4)
    fig.suptitle(title, fontsize=20)

    for data in dataSet:
        ax.plot(data['x'], data['y'], label=data['label'])
        yLimData = cp.copy(data['y'])
        yLimData[np.isinf(yLimData)] = 0
        yLims = \
            np.vstack((yLims, [np.nanmin(yLimData), np.nanmax(yLimData)]))
        xLims = \
            np.vstack((xLims, [data['x'][0], data['x'][-1]]))
    
    ax.set_xlabel(xLabel, fontsize=20)
    ax.set_ylabel(yLabel, fontsize=20)
    ax.legend(loc='best', fontsize=12)

    if xLim is None:
        xLim = [np.nanmin(xLims[:,0]), np.nanmax(xLims[:,1])]
    ax.set_xlim(xLim)

    if yLim is None:
        yLim = [np.nanmin(yLims[:,0]), np.nanmax(yLims[:,1])]
        margin = (yLim[1] - yLim[0]) / 20
        yLim = [yLim[0]-margin, yLim[1]+margin]
    ax.set_ylim(yLim)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    return fig

def freq(dataSet, smooth, xLabel, yLabel, yLim, xLim, title, decimalSep):
    """Plot the signal decibel magnitude in frequency domain.

    Parameters (default), (type):
    -----------------------------

        * dataSet (), (list):
            a list with dictionaries containing the information about
            each curve. Needed keys for frequency plot:

            - 'x', (), (ndarray): time axis;
            - 'y', (), (ndarray): amplitude axis;
            - 'label', (), (str): curve label;
            - 'dBRef', (), (float): decibell reference;
            - 'minFreq', (), (float): SignalObj's minimum analysed frequency
            - 'maxFreq', (), (float): SignalObj's maximum analysed frequency

            >>> dataSet = [{'x':x, 'y':y, 'label':'my beautiful curve,
                            'dBRef':2e-5, 'minFreq':20, 'maxFreq':20000}]
        
        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Amplitude'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * xLim (), (list):
            left and right limits

            >>> xLim = [15, 21000]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

    Return:
    --------

        matplotlib.figure.Figure object.
    """
    if decimalSep == ',':
        locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
        plt.rcParams['axes.formatter.use_locale'] = True
    elif decimalSep =='.':
        locale.setlocale(locale.LC_NUMERIC, 'C')
        plt.rcParams['axes.formatter.use_locale'] = False
    else:
        raise ValueError("'decimalSep' must be the string '.' or ','.")

    if xLabel is None:
        xLabel = 'Frequency [Hz]'
    if yLabel is None:
        yLabel = 'Magnitude'
    if title is None:
        title = ''

    yLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.10, 0.21, 0.88, 0.72], polar=False,
                        projection='rectilinear', xscale='linear')
    ax.set_snap(True)
    ax.grid(color='gray', linestyle='-.', linewidth=0.4)
    fig.suptitle(title, fontsize=20)

    for data in dataSet:
        dBSignal = 20*np.log10(data['y']/data['dBRef'])

        if smooth:
            dBSignal = ss.savgol_filter(dBSignal, 31, 3)
        
        ax.semilogx(data['x'], dBSignal, label=data['label'])

        yLimData = dBSignal
        yLimData[np.isinf(yLimData)] = np.nan
        yLims = \
            np.vstack((yLims, [np.nanmin(yLimData), np.nanmax(yLimData)]))

    ax.set_xlabel(xLabel, fontsize=20)
    ax.set_ylabel(yLabel, fontsize=20)
    ax.legend(loc='best', fontsize=12)

    if xLim is None:
        xLim = [data['minFreq'], data['maxFreq']]
    ax.set_xlim(xLim)

    if yLim is None:
        yLim = [np.nanmin(yLims[:,0]), np.nanmax(yLims[:,1])]
        margin = (yLim[1] - yLim[0]) / 20
        yLim = [yLim[0]-margin, yLim[1]+margin]
    ax.set_ylim(yLim)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(2,5),
                                                 numticks=5))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    return fig

def bars(dataSet, xLabel, yLabel, yLim, title, decimalSep, barWidth,
         errorStyle):
    """Plot the analysis data in fractinal octave bands.

    Parameters (default), (type):
    -----------------------------

        * dataSet (), (list):
            a list with dictionaries containing the information about each
            curve. Needed keys for analysis plot:

            - 'bands', (), (ndarray): fractional octave bands;
            - 'data', (), (ndarray): magnitude axis;
            - 'dataLabel', (), (str): data label;
            - 'error', (), (ndarray): error for each fractional octave band;
            - 'errorLabel', (), (str): error label;
            
            >>> dataSet = [{'bands':[100, 200, 400], 'data':[1, 5 ,7],
                            'dataLabel':'my precious data,
                            'error':[2e-5, 1, 3],
                            'errorLabel':'my mistakes'}]
        
        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Amplitude'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

        * barWidth (0.75), float:
            width of the bars from one fractional octave band. 
            0 < barWidth < 1.

        * errorStyle ('standard'), str:
            error curve style. May be 'laza' or None/'standard'.

    Return:
    --------

        matplotlib.figure.Figure object.
    """
    if errorStyle == 'laza':
        ecolor='limegreen'
        elinewidth=20
        capsize=0
        capthick=0
        alpha=.60
    else:
        ecolor='black'
        elinewidth=5
        capsize=10
        capthick=5
        alpha=.60

    if decimalSep == ',':
        locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
        plt.rcParams['axes.formatter.use_locale'] = True
    elif decimalSep =='.':
        locale.setlocale(locale.LC_NUMERIC, 'C')
        plt.rcParams['axes.formatter.use_locale'] = False
    else:
        raise ValueError("'decimalSep' must be the string '.' or ','.")

    if xLabel is None:
        xLabel = 'Frequency bands [Hz]'

    if yLabel is None:
        yLabel = 'Modulus'
    
    if title is None:
        title = ''
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.10, 0.21, 0.88, 0.72], polar=False,
                        projection='rectilinear', xscale='linear')
    ax.set_snap(True)
    ax.set_xlabel(xLabel, fontsize=20)
    ax.set_ylabel(yLabel, fontsize=20)
    plt.title(title, fontsize=20)

    yLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)

    dataSetLen = len(dataSet)
    
    # checking negative plot necessity
    minVal = np.inf
    negativeCounter = 0
    for data in dataSet:
        # minimum value
        newMinVal = np.amin(data['data'])
        marginData = \
            [value for value in data['data'] if not np.isinf(value)]
        margin = np.abs((np.nanmax(marginData) -
            np.nanmin(marginData)) / 20)
        newMinVal = newMinVal - margin
        newMinVal += np.sign(newMinVal)
        if newMinVal < minVal:
                minVal = newMinVal
        # negative counter
        for value in data['data']:
            if value < 0:
                negativeCounter += 1

    for dtIdx, data in enumerate(dataSet):
        errorLabel = \
            'Error' if data['errorLabel'] is None else data['errorLabel']
        label = \
            '' if data['dataLabel'] is None else data['dataLabel']

        fbar = np.arange(0,len(data['data']))

        if negativeCounter < (len(data['data'])*dataSetLen)//2:
            minVal = 0
        
        ax.bar(fbar + barWidth*dtIdx/dataSetLen,
            -minVal + data['data'],
            width=barWidth/dataSetLen, label=label, zorder=-1)

        if data['error'] is not None:
            ax.errorbar(fbar + barWidth*dtIdx/dataSetLen,
                    -minVal + data['data'],
                    yerr=data['error'], fmt='none',
                    ecolor=ecolor,
                    elinewidth=elinewidth*barWidth/dataSetLen,
                    capsize=capsize*barWidth/dataSetLen,
                    capthick=capthick*barWidth/dataSetLen,
                    zorder=0,
                    fillstyle='full',
                    alpha=alpha,
                    barsabove=True,
                    label=errorLabel)

        error = data['error'] if data['error'] is not None else 0

        ylimInfData = -minVal + data['data'] - error
        ylimInfData = \
            [value for value in ylimInfData if not np.isinf(value)]
        ylimInfMargin = \
            np.abs((np.nanmax(ylimInfData) - np.nanmin(ylimInfData)) / 20)
        ylimInf = np.nanmin(ylimInfData) - ylimInfMargin
    
        ylimSupData = -minVal + data['data'] + error
        ylimSupData = \
            [value for value in ylimSupData if not np.isinf(value)]
        ylimSupMargin = \
            np.abs((np.nanmax(ylimSupData) - np.nanmin(ylimSupData)) / 20)
        ylimSup = np.nanmax(ylimSupData) + ylimSupMargin

        yLims = \
            np.vstack((yLims,
                        [ylimInf, ylimSup]))

    if yLim is None:
        yLim = [np.nanmin(yLims[:,0]), np.nanmax(yLims[:,1])]
    ax.set_ylim(yLim)
    ax.autoscale(enable=True, axis='x', tight=True)
    
    ax.grid(color='gray', linestyle='-.', linewidth=0.4)

    ax.set_xticks(fbar+barWidth*(dataSetLen-1)/dataSetLen-
        barWidth*(dataSetLen-1)/(2*dataSetLen))
    xticks = data['bands']
    ax.set_xticklabels(['{:n}'.format(tick) for tick in xticks],
                        rotation=45, fontsize=14)

    def neg_tick(x, pos):
        return '%.1f' % (x + minVal if x != minVal else 0)
    formatter = ticker.FuncFormatter(neg_tick)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
    ax.yaxis.set_major_formatter(formatter)

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)        
    
    ax.legend(loc='best', fontsize=13, framealpha=0.6)
    
    return fig
