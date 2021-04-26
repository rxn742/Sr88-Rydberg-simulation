# Sr88-Rydberg-simulation
GUI simulation of 3 level singlet and triplet excitation pathways to Rydberg states in Strontium-88, with the capabilities of plotting steady state population probabilities and probe laser transmission, allowing observation of quantum interference effects such as electromagnetically induced transparency.

This is coded by Robert Noakes and part of a 4th year undergraduate masters project between Robert Noakes and Louis-Miranda Smedley investigating Rydberg-Stark deceleration for use in a Strontium optical lattice clock.

## Installation

If you have git installed, download the repository via
```bash
git clone https://github.com/rxn742/Sr88-Rydberg-simulation.git
```
Another option is to download the repository as a .zip file and then extract to your desired location.

This simulation has several dependencies that can be installed using a chosen package manager. If using Anaconda, cd into the main directory and install the virtual environment for your OS.

Windows
```bash
conda env create -f GUI_Windows.yml
```

Mac
```bash
conda env create -f GUI_Mac.yml
```

Linux
```bash
conda env create -f GUI_linux.yml
```

If using pip, install the requirements via

```bash
pip install -r requirements.txt
```
You will also need to have installed a C++ compiler to compile code used by cython. This will usually come as standard with linux distributions and OSX, but if you do not have one, please install a compiler such as 

GCC 4.7+ or MS VS 2015

## Usage

When all requirements are installed, cd into the /GUI directory and run the GUI by

```bash
python GUI.py
```

There you will be greeted with the main page of the GUI with all options availible. Please consult the wiki for instructions on using the GUI.

## Contributing

For any bugs or update requests please raise an issue.

