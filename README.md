# ASPSIM : Audio signal processing simulator

## Introduction
The package simulates sound in a reverberant space. The simulator supports moving microphones and moving sources. The simulator supports adaptive processing, where the sound of the sources are determined by the sound in the room. This is useful for instance for active noise control and similar adaptive processing tasks. 

The package currently uses the image-source method implementation of [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics) to generate the room impulse responses. 

**[More info and complete API documentation](https://sounds-research.github.io/aspsim/)**


## Installation
The package can be installed with pip as
```
pip install aspsim
```

Alternatively, clone the repository and install the package direct from the source code as
```
pip install path/to/aspsim
```


## Usage
A number of examples with varying complexity can be found in the examples folder. 
