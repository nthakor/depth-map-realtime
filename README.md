# Fast feature matching using GPU in video.
The repository is part of the project course ES 611 Programming Advanced Computer Architectures. We are trying to implement feature matching using KDTree on GPU. 

We are using following papers:
* Popov, Stefan, et al. "Stackless KD‐Tree Traversal for High Performance GPU Ray Tracing." Computer Graphics Forum. Vol. 26. No. 3. Blackwell Publishing Ltd, 2007.
* Zhou, Kun, et al. "Real-time KD-tree construction on graphics hardware." ACM Transactions on Graphics (TOG) 27.5 (2008): 126.

Both the papers are uploaded in the repository.

Task Done:
* Written a python code to take input from web-cam, find feature descriptors and display key-points

To view run vread.py 
Required Packages:
* OpenCV
* Numpy

Detailed Project Description:
* Find feature descriptors of a single frame
* Create KDTree of the first frame descriptors
* For each descriptors for next frames check whether nearest neighbor exist. 
* Create new KDTree of descriptors with nearest neighbor
* Repeat this process till end.

Task to be performed:
* Create a function which can track the changing position of feature. 
* Develop running serial code in C++
* Implement parallel version of KDTree using above mentioned papers. 
* If time permits implement a parallel version of feature descriptors using following paper.

Parallel SIFT paper:

Acharya, Aniruddha, and R. Venkatesh Babu. "Speeding up SIFT using GPU." Computer Vision, Pattern Recognition, Image Processing and Graphics (NCVPRIPG), 2013 Fourth National Conference on. IEEE, 2013.