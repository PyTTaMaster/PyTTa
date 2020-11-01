## PyTTa (Python in Technical Acoustics) v0.1.0 beta 9

The project began as an effort to create audio, acoustics and vibrational data
acquiring and analysis toolbox to a free cost level, high-end results,
combining the passion for programming with the expertise in acoustics and
vibration of the Acoustics Engineers from the Federal University of Santa Maria.

We are students, teachers, engineers, passionates and inquiring people,
on the first steps of a journey throughout the Python path to bring Acoustics
to the Open Seas, Open Sources and Open World!

### Usage

We strongly recommend using Anaconda Python Distribution, as it integrates an
IDE (Spyder) and lots of packages over which PyTTa is based on as Numpy, Scipy, 
Matplotlib and PyPI (pip) which is used to install Sounddevice and PyTTa itself.

The toolbox offers some main classes intended to do measurements (Rec, PlayRec
and FRFMeasurement), handle streaming functionalities (Monitor and Streaming),
handle signals/processed data (SignalObj and Analysis), filter (OctFilter),
communicate with some hardware (LJU3EI1050), and also new features whose should 
have an object-oriented implementation.

There are some assistant functions intended to manipulate and visualize signals
and analyses stored as SignalObj and Analysis objects. These functions are
called from the toolbox's top level.

The top level modules offers tools which receives as arguments and returns
PyTTa objects. Utilities comprises signal synthesis (pytta.generate), room
acoustics parameters calculation (pytta.rooms), and some other calculations
according to standards (pytta.iso3741).

The sub-packages contains some expansion of the toolbox, as general utilities
(pytta.utils) and applications built from the toolbox's basic functionalities
(pytta.roomir).

PyTTa is now a multi-paradigm toolbox, which may change in the future. We aim
from now on to be more Pythonic, and therefore more objected-oriented.

### Documentation

This package aims to be an easy interface between acoustician and vibrational
engineers in the use of Python for study, engineering or any other ends that
it may suite. From import to each function and attribute we wrote a little
documentation that may be useful for understanding everything available,
and what is just a means to an end.

To begin, one can try:

    >>> import pytta
    >>> pytta.list_devices()
    >>> pytta.default()

This set of commands will print the available audio I/O devices and the
default parameters as they are on the default class object.

To read everything available on the package, and assuming the use of
Spyder IDE, one can press "ctrl+i" with the cursor in front of the module,
submodule, class, methods, or function names; this will open the help menu
with the documentation of the respective item.
    
    >>> pytta|
    >>> pytta.properties|
    >>> pytta.generate|
    >>> pytta.functions|
    >>> pytta.classes|
    >>> pytta.apps|
    >>> pytta.utils|

The | represents the cursor position to press "ctrl+i" to use the Spyder help
widget.

Inside each submodule, the user will find instructions on the available tools,
and how to access them.

The documentation is also available at Read The Docs:
https://pytta.readthedocs.io/

### Dependencies

- Numpy;
- Scipy;
- Matplotlib;
- Sounddevice;
- Soundfile;
- H5py;
- Numba.

### Installation

To install the last version compiled to pip, which can be slightly behind of
development branch, do:

    >>> pip install pytta
    
If you want to check the most up to date beta version, please get the
development branch source code, clone our repository to your local git, or
even install it from pip, as follows:

    >>> pip install git+https://github.com/pyttamaster/pytta@development

### Contributing as a user

Using the toolbox, you are already contributing. If you experience any bugs,
have any suggestion or idea, have anything to say about it, please consider
sending us a message.

### Contributing as a developer

Please read our [CodingGuidelines](https://github.com/PyTTAmaster/PyTTa/blob/development/CodingGuidelines) and [Contributing](https://github.com/PyTTAmaster/PyTTa/blob/development/Contributing.md) files.

### Contact

Contact us at pytta@eac.ufsm.br .

## More information

[Main Website](https://sites.google.com/eac.ufsm.br/pytta/)

[Acoustical Engineering UFSM Website](http://www.eac.ufsm.br)

[UFSM Website](https://www.ufsm.br)

[Download Spyder (with Anaconda)](https://www.anaconda.com/download/)

[Or Miniconda](https://conda.io/en/latest/miniconda)
