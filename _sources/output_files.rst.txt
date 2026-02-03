Output files from simulator
===========================
The simulator will generate a new folder and output a number of files there, as long as a folder is provided when creating the SimulatorSetup object, and the sim_info.plot_output is set to any other valid option than "none". 

For most usecases, there are other signals or objects that are of interest, and the user can then specify what should be recorded and saved to file. For more information on the user-selected outputs, see the documentation 
on logging. 



array_pos.json
--------------
A file containing the positions of the array elements in meters. 

array_pos.pdf
--------------
A plot of the array positions in the xy-plane. The z-direction is not visible. If any of the paths are generated using the image-source method, the walls of the room is also included in the figure. 


config.yaml
--------------
A copy of the config used to run the simulation. Corresponds to the sim_info object.

metadata_arrays.json
-----------------------
Contains information about the arrays used in the simulation. The fields listed below are included for all array types. For some types of arrays, such as FreeSourceArrays and RegionArray there is additional information as well. 

:type:  
    Which type of array it is. 
:number:  
    How many array elements there are in the array. 
:numberofgroups: 
    The number of groups in the array. Currently the group feature is undocumented, and could be removed in the future.
:dynamic:  
    Whether the array is dynamic, meaning has a time-varying position or not. 


metadata_paths.json
-----------------------
Contains information about the acoustic paths between sources and microphones. For each pair of source and microphone array there will be an entry in the form of source_name->mic_name, containing different entries depending on path type. The following is always incldued

:type: Which propagation type was chosen for the path. 

metadata_processor.json
-----------------------
Contains information about the processors used in the simulation. If no processor is used, then this file will not be generated. There is no required information in the file, but can be populated with relevant information by the user. The baseclass Processor has a dictionary named metadata, and therefore any user-specified processor will do as well. The contents of that dictionary will be saved as this file.