# MRI-scan-segmentation
## Multi-level brain tumour segmentation in MRI scans using Nature inspired algorithms
Nature enthused algorithms are the most potent for optimization. Several bio-inspired algorithms were developed to generate optimum threshold values for segmenting such images efficiently. Their exhaustive search nature makes them computationally expensive when extended to multilevel thresholding. In this project, we propose a computationally efficient image segmentation algorithm, called **CSMcCulloch**, incorporating McCulloch’s method for levy flight generation in **Cuckoo Search** (CS) algorithm to optimise multi level thresholding. In addition to it, we have also implemented the **Ant Colony algorithm optimisation** for **Fuzzy C Means** to obtain optimum level image segmentation of brain tumor from the Magnetic Resonance Images (MRI).
This project explores the comparison between the two, performing a profound study of their search mechanisms to discover how it is efficient in detecting tumors and compare their respective experimental results.

##### Our code for Segmenting Gray/RGB image using a modified Cuckoo Search algorithm (CSMcCulloch) tested with different objective functions. 
**CSMC_otsu.m** : The function which can be run to view a sample result of segmenting a gray or RGB image using CS MCulloch algorithm with Otsu's between class variance as objective function 
**CSMC_kapur.m** : The function which can be run to view a sample result of segmenting a gray or RGB image using CS MCulloch algorithm with Kapur's entropy as objective function 
**CSMC_tsallis.m** : The function which can be run to view a sample result of segmenting a gray or RGB image using CS MCulloch algorithm with Tsallis entropy as objective function .

##### Programming Constructs used:
No of thresholds for all cases = 5
Software: Matlab 2015a
Hardware: i5 processor, 4GB RAM
Processing time: 
- a) Otsu’s Method: 9.554s
- b) **Kapur’s Entropy: 6.247s**
- c) Tsallis Entropy: 7.677s

Profiles of each code, function-wise with processing time has been uploaded as:
- a) Otsu’s Method: prof_otsu.png
- b) Kapur’s Entropy: prof_kapur.png
- c) Tsallis Entropy: prof_tsallis.png
- d) ACO Algo: prof_aco.png
