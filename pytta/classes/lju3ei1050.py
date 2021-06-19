# -*- coding: utf-8 -*-

# TO DO: rename this module to labjack and merge ei1050.py. Then adjust 
#        roomir examples.

import sys
from queue import Queue

try:
    import LabJackPython
    import u3
    from . import ei1050
except:
    raise ImportError('Driver error - The driver could not be imported.\n\
                      Please install the UD driver (Windows) or Exodriver (Linux and Mac OS X) from www.labjack.com')
    
class main():
    """
    Class for communication with the Labjack U3 hardware with the probe EI1050
    to acquire temperature and relative humidity. 
    
    The UD driver (Windows) or Exodriver (Linux and Mac OS X) from
    www.labjack.com must be installed.
    
    Methods:
    ---------
    
        * start():
            start communication;
        
        * stop():
            stop communication;
            
        * read():
            gets the latest reading from the readings queue and display it;
            
        * instructions():
            show information for connections between the EI1050 probe and the 
            Labjack U3hardware;
            
    """    
    def __init__(self):
        # Ensure the existence of a thread, queue, and device variable
        self.targetQueue = Queue()
        self.thread = None
        self.device = None

        # Determine if we are reading data
        self.reading = False

    def start(self):
        """
        Name:main.start()
        Desc:Starts reading values from EI1050 probe
        """
        try:
            # Get device selection
            if len(LabJackPython.listAll(3)) > 0:
                self.device = u3.U3()

#            print('Device serial number : '+str(self.device.serialNumber))

            # Create and start the thread
            self.thread = ei1050.EI1050Reader(self.device, self.targetQueue)

            # Start scheduling

            self.thread.start()

        except:
            showError(sys.exc_info()[0], sys.exc_info()[1])

    def stop(self):
        """
        Name:main.stop()
        Desc: Stops reading values from EI1050 probe
        """
        self.thread.stop()
        self.thread.join()
        self.device.close()

    def read(self):
        """
        Name:main.read()
        Desc: Gets the latest reading from the readings queue and display it
        """
        # Check for errors
        if self.thread.exception is not None:
            showError(self.thread.exception[0], self.thread.exception[1])
        else:
            # Get the actual values
            latestReading = None
            while not self.targetQueue.empty():
                latestReading = self.targetQueue.get()

            if latestReading is not None:
                latestReading.getTemperature()
                latestReading.getHumidity()
                latestReading.getStatus()

        return (latestReading.getTemperature(),latestReading.getHumidity())

    def instructions(self):
        print("Instructions")
        print('U3 SHT configured with pins as follows:\
            \nGreen(Data) -- FIO4\
            \nWhite(Clock) -- FIO5\
            \nBlack(GND) -- GND\
            \nRed(Power) -- FIO7\
            \nBrown(Enable) -- FIO7')


def showError(title, info):
        """
        Name:showError()
        Desc:Shows an error popup for last exception encountered
        """
        print(title + ' - ' + str(info) + "\n\nPlease check your " +
                               "wiring. If you need help, click instructions.")