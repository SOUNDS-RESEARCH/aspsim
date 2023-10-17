# ASPSIM : Audio signal processing simulator

## Introduction
The package simulates sound in a reverberant space. The simulator supports moving microphones and moving sources. The simulator supports adaptive processing, where the sound of the sources are determined by the sound in the room. This is useful for instance for active noise control and similar adaptive processing tasks. 

The package currently uses the image-source method implementation of [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics) to generate the room impulse responses. 


## Installation
Requires python version that satisfies 3.8 <= version <= 3.11, as those are supported by Numba.
### Quick installation
1. Install ASPCORE by following the instructions at https://github.com/SOUNDS-RESEARCH/aspcore/tree/develop (including ASPCORE installation via `pip install -e ./aspcore`)\
2. Navigate into an appropriate folder and run
```
git clone https://github.com/SOUNDS-RESEARCH/aspsim.git
pip install ./aspsim
pip install -r aspsim/requirements.txt
```

### Installing the package
The package can be installed by opening your favorite terminal and writing (changing out cd for the appropriate change-directory command, and the path to whatever you want)
```
cd c:/folder/parent_folder_to_code
git clone https://github.com/SOUNDS-RESEARCH/aspsim.git
pip install ./aspsim
```

If you intend to work on the development, it might be preferable to get the develop branch directly instead, and to use an editable install. Use the commands
```
cd c:/folder/parent_folder_to_code
git clone --branch develop https://github.com/SOUNDS-RESEARCH/aspsim.git
pip install -e ./aspsim
```

### Dependencies
To install all (except aspcore) dependencies use the command
```
pip install -r requirements.txt
```

It is required to have [aspcore](https://github.com/SOUNDS-RESEARCH/aspcore) installed, which does not exist on PYPI as of yet (i.e. you can not install with pip install the regular way). \
The collections modules, namely diagnostics/diagnosticscollection.py and signal/sourcescollection.py, are allowed to depend on the package [aspcol](https://github.com/SOUNDS-RESEARCH/aspcol). It is only required if you make use of classes or functions from a collections module. 


## Usage
A number of examples with varying complexity can be found in the examples folder. 

sim.add_processor(proc)
sim.run_simulation()
```
