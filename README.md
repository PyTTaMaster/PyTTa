<p>
<a href="https://github.com/PyTTAmaster/PyTTa/blob/development/LICENSE">
  <img src="https://img.shields.io/github/license/pyttamaster/pytta" alt="PyTTa License" />
</a>

<img src="https://img.shields.io/pypi/pyversions/pytta" alt="PyTTa Python supported versions" />

<a href="https://pypi.org/project/PyTTa/#history">
  <img src="https://img.shields.io/pypi/v/pytta" alt="PyTTa PyPI version" />
</a>

<img src="https://img.shields.io/pypi/dm/pytta" alt="PyTTa PyPI downloads" />

<a href="https://github.com/pyttamaster/pytta/actions?query=workflow:(If%20commit%20contains%20version%20tag)%20Build%20and%20publish%20PyTTa%20distribution%20package%20to%20PyPI">
  <img src="https://img.shields.io/github/workflow/status/pyttamaster/pytta/(If%20commit%20contains%20version%20tag)%20Build%20and%20publish%20PyTTa%20distribution%20package%20to%20PyPI" alt="PyTTa build workflow status" />
</a>
</p>


## PyTTa (Python in Technical Acoustics)

The project began as an effort to create audio, acoustics, and vibration data acquiring and analysis toolbox to a free cost level, high-end results, combining the passion for programming with the expertise in Acoustics and Vibration of the Acoustics Engineers from the Federal University of Santa Maria (Southern Brazil).

We are students, teachers, engineers, passionate and inquiring people, on the first steps of a journey throughout the Python path to bring Acoustics to the Open Seas, Open Sources, and Open World!

### Usage

We strongly recommend using Anaconda Python Distribution, as it integrates an IDE (Spyder) and lots of packages over which PyTTa is based on as Numpy, Scipy, Matplotlib, and PyPI (pip). The latter is used to install Sounddevice and PyTTa itself. 

The toolbox offers some main classes intended to do measurements (Rec, PlayRec, and FRFMeasurement), handle streaming functionalities (Monitor and Streaming), handle signals/processed data (SignalObj and Analysis), filter (OctFilter), communicate with some hardware (LJU3EI1050), and also new features containing an object-oriented implementation. 

There are some assistant functions intended to manipulate and visualize signals and analyses stored as SignalObj and Analysis objects. These functions are called from the toolbox's top level.

The top-level modules offer tools that receive as arguments and return PyTTa objects. Utilities comprise signal synthesis (pytta.generate), room acoustics parameters calculation (pytta.rooms), and some other calculations according to standards (pytta.iso3741).

The sub-packages contains some expansion of the toolbox, as general utilities (pytta.utils) and applications built from the toolbox's basic functionalities (pytta.roomir).

PyTTa is now a multi-paradigm toolbox, which may change in the future. We aim from this point on to be more Pythonic, and therefore, more objected-oriented.  

### Documentation

This package aims to be an easy interface between acousticians and vibrational engineers in the use of Python for study, engineering, or any other ends that it may suit. From import to each function and attribute we wrote a little documentation that may be useful for understanding everything available, and what is just a means to an end.

To begin, one can try:

    >>> import pytta
    >>> pytta.list_devices()
    >>> pytta.default()

This set of commands will print the available audio I/O devices and the
default parameters as they are on the default class object.

To read everything ready for use on the package, and assuming the use of Spyder IDE, one can press "ctrl+i" with the cursor in front of the module, submodule, class, methods, or function names; this will open the help menu with the documentation of the respective item. 
    
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
- H5py; and
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

Using the toolbox, you are already contributing. If you experience any bugs, have any suggestions or ideas, have anything to say about it, please consider sending us a message.

### Contributing as a developer

Please read our [CodingGuidelines](https://github.com/PyTTAmaster/PyTTa/blob/development/CodingGuidelines) and [Contributing](https://github.com/PyTTAmaster/PyTTa/blob/development/Contributing.md) files.

### Contact

Contact us at pytta@eac.ufsm.br .

### Acknowledge 

If you use PyTTA in your research, work, paper, or university assignment, acknowledge and cite us if possible. 🙏

 - PyTTa – Python in Technical Acoustics (GitHub). https://github.com/pyttamaster/pytta, 2020.
 - W. D’A. Fonseca, J. V. Paes, M. Lazarin, M. V. Reis, P. H. Mareze, and E. Brandão. *PyTTa: Open source toolbox for acoustic measurement and signal processing*. In 23 International Congress on Acoustics - ICA 2019, pages 1–8, Aachen, Germany, 2019. doi: [10.18154/RWTH-CONV-239980](http://doi.org/10.18154/RWTH-CONV-239980).
 - Research Gate Publication Page https://www.researchgate.net/project/PyTTa-Python-in-Technical-Acoustics

BibTex file [PyTTa.bib](https://github.com/PyTTAmaster/PyTTa/blob/development/docs/PyTTa.bib)

## More information

<!--[Main Website](https://sites.google.com/eac.ufsm.br/pytta/) -->

[Acoustical Engineering UFSM Website](http://www.eac.ufsm.br)

[UFSM Website](https://www.ufsm.br)

[Download Spyder (with Anaconda)](https://www.anaconda.com/download/)

[Or Miniconda](https://conda.io/en/latest/miniconda)
