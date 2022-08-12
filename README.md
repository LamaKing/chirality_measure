# Chirality measure

Measure the chirality of 2D pattern. 
The measure is implemented as the minimum of the cross correlation between the original pattern and its x-inverted one, rotated on a given array of angles.
For details see http://paulbourke.net/miscellaneous/correlate/

## Requirments

- Python3 3.7.11
- NumPy 1.21.6
- Matplotlib 3.3.0
- ASE 3.19.2
- SciPy 1.5.2

For the notebook example:
- Jupyter notebook 6.1.1
- IPython 6.5.0 

## Usage

### As Python package
Import the function chirality_xcorellation from chirality_xcorell.
The source needs to be in the same folder of in the python path (check sys.path variable in your python environment)
See the notebook Example_chirality_measure and the source for details

### On XYZ trajectory from CLI
To obtain the chirality of each frame in a xyz trajectory use
./chirality_xcorell.py traj.xyz :

To obtain the chirality of a speficic frame in a xyz trajectory use
./chirality_xcorell.py traj.xyz <n_frame>

They will print the chirality of each frame on standard output.
