# Introduction
Many fields in research and industry struggle with having too much data and too little storage. At Large Hadron Collider experiments like ATLAS, the disk space needed in 5 years is 10 times more than the projected available disk space with projected funding and technology trends.

Data formats used in big data fields are already very optimized for storage. The ROOT data format used in particle physics does not compress under normal loss-less compression methods like zip. Luckily, the observable in particle physics are often statistical variables, where the number of observed events is key to finding new physics.

Since loss-less compression doesn’t provide more storage, and  the observable benefit from more events, lossy compression is a good alternative. Using lossy compression some data accuracy will be lost, but the compression will allow for more events to be stored which will increase the statistical precision.

![](./images/storage_need.png "Estimated disk-space requirements of the ATLAS experiment from 2018 to 2028 compared to a flat-funding scenario with an increase of 20% per year based on the current technology trends.")

