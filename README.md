# Sr88-Rydberg-simulation
GUI simulation of 3 level singlet and triplet excitation pathways to Rydberg states in Strontium-88, with the capabilities of plotting steady state population probabilities and probe laser transmission, allowing observation of quantum interference effects such as electromagnetically induced transparency.

This is coded by Robert Noakes and part of a 4th year undergraduate masters project between Robert Noakes and Louis-Miranda Smedley investigating Rydberg-Stark deceleration for use in a Strontium optical lattice clock.

## Installation

This simulation has several dependencies that can be installed using a chosen package manager. If using anaconda, choose the virtual environment for your OS and install

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

If using pip, install the requirements

```bash
pip install -r requirements.txt
```

## Usage

When all requirements are installed, cd to the /GUI directory and run the GUI by

```bash
python GUI.py
```
