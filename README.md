![](images/routes.png)

PtX transport is a multi-commodity multi mean of transport algorithm, capable to find the most cost-effective transportation route from pre-defined starting points to a desired location. The algorithm derives possible solutions for transportation from a starting point to the final destination. Based on the underlying infrastructure data, the algorithm iteratively explores infrastructure nodes and calculates costs for each reached nodes. It terminates as soon as it reaches the final destination and exploration to other nodes is not possible anymore.

![Graphical abstract of the approach](images/graphical_abstract.png)

To use the model, following project structure needs to be created.

    project folder/
    |   
    +---data/
    |   +---raw data/
    |   +---processed data/
    |   
    +---results/

If processed data is already available, it can be placed directly in the processed data folder (the example data folder provides processed data for demonstration purposes). Otherwise, data for oil and gas infrastructure as well as ports can be downloaded and used (e.g., oil & gas pipelines: Global Energy Monitor | ports: World Port Index). This data needs to be placed in the raw data folder for further processing.

To run the algorithm, 

Citation
Big thanks to
