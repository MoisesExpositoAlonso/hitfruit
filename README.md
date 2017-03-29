
# hitfruit: skeletonisation of plant inflorescences' images for fruit estimation
## Moises Exposito-Alonso (moisesexpositoalonso@gmail.com)


## Usage
The code below will download the repository and run the example dataset. The output
images will be under the folder /sk and be labeled as "proc" for the segmented image,
"sk" for the skeletonised image, and "end" or "branches" for the branching or end points
overlayed onto the skeletonised image. A tab separated file, .tsv, is produced to store the number of non-black pixels of the processed images.

``` sh
git clone https://github.com/MoisesExpositoAlonso/hitfruit
cd hitfruit 

python exampleskeleton.py

```

### NOTE

This module depends on functions from my other [hippo](https://github.com/MoisesExpositoAlonso/hippo) and [mpy](https://github.com/MoisesExpositoAlonso/moi-py) repositories. So two lines in the scripts runhitfruit.py and hitfruit.py have to be modified indicating the path to the hippo and moi modules in your computer.
