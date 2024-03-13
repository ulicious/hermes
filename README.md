![](images/routes.png)

PtX transport is a multi-commodity multi mean of transport algorithm, capable to find the most cost-effective transportation route from pre-defined starting points to a desired location. The algorithm derives possible solutions for transportation from a starting point to the final destination. Based on the underlying infrastructure data, the algorithm iteratively explores infrastructure nodes and calculates costs for each reached nodes. It terminates as soon as it reaches the final destination and exploration to other nodes is not possible anymore.

<p align="center">
  <img src="images/graphical_abstract.png" />
</p>

Following steps need to be taken to use PtX Transport:
-

1. Download this repository and create python project in your coding environment
2. Install requirements
3. Create following directory structure


    project folder/
    |
    +---raw data/
    +---processed data/
    +---results/


5. Adjust "configuration.yaml" (paths, parameters etc.)
6. Optional: Place your own raw or processed data in the respective folder. This data must resemble the data in the raw and processed data folder in the github repository
6. If you need to process raw data, run "data_processing/script_process_raw_data.py"
7. If you want to start the routing algorithm, run "main.py"

Important things to consider:
-
- Data processing is quite time-consuming and heavily depends on the resources of you computer
- The processed data will take quite some storage space

Citation
Big thanks to
