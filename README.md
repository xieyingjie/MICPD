# MICPD
### Multi-view Change Point Detection in Dynamic Networks

## Abstract
Change point detection aims to find the locations of sudden changes in the network structure, which persist with time.
However, most current methods usually focus on how to accurately detect change points, without providing deeper insight into the cause of the change.
In this study, we propose the multi-view feature interpretable change points detection method (MICPD), which is based on a vector autoregressive(VAR) model to encode high-dimensional network data into a low-dimensional representation, and locate change points by tracking the evolution of multiple targets and their interactions across the whole timeline. 
According to the evolutionary nature of dynamic networks, we define a categorization of different types of changes which can occur in dynamic networks.
We compare the performance of our method with state-of-the-art methods on four synthetic datasets and the world trade dataset. Experimental results show that our method achieves well in most cases.

**The baselines folder contains baselines for all comparisons of the experiment.**

**Run the code**


Please set the root directory of the project as your Python path.

For Synthetic dataset

```bash
python Synthetic/SBM_generator.py

community Tracker
```bash
python communityTracker/Tracker.py  <inputfile.json>

```bash
python main.py  --datapath="path"


