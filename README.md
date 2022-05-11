# End-to-end-GNN-based-recommender-system

Table of contents:
- [Project description](#project-description)
- [Related work](#related-work)
- [Tags](#tags)
- [Data overview & graph structure](#data-overview-and-graph-structure)
- [Approach](#approach)
- [Results](#results)
- [Credits](#credits)

## Project description
The classical approach to recommender systems is to use either User-to-User or Item-to-Item Collaborative Filtering, or a combination of the two. These approaches have been extensively covered from both research and business use case angles. Next, graph-based approach have been proposed [[1]](https://www.sciencedirect.com/science/article/abs/pii/S0167923612002540). However, they typically employed a multi-stage pipeline, consisting of separate graph feature extraction models and link prediction models, which were treated and trained separately. In this project we explore an end-to-end solution, which contrary to those early approaches trains the entire pipeline as one piece.

<p align="center">
  <img width="500" src=https://user-images.githubusercontent.com/74935134/164030033-a81cf0ca-4206-4a16-856b-7d21a00ddac3.jpeg>
</p>

## Related work
- [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) - This paper tackles a similar problem to ours and serves as a performance benchmark. However, we use different data and a more detailed architecture.
- [The Netflix Recommender System: Algorithms, Business Value, and Innovation](https://dl.acm.org/doi/10.1145/2843948) - Provides us with an overview of different approaches to recommender problem and their effectiveness
- [Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108) - This paper shows an improvement over traditional Collaborative Filtering by introducing the collaborative signal into the GNN embedding process 

## Tags
Grahp Neural Networks, Recommender System, Graph Attention Network, Heterogenous Graphs, Bipartitie Graphs, Pytorch Geometric

## Data overview and graph structure
Data used for this project comest from [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). From the data we generated user and movie features (sample can be seen in the data folder) and utilized ratings as edge weights. Having limited processsing power we used a subsample of the data:

`Number of users: 671`  
`Number of movies: 2816`  
`Number of ratings: 44905`

There are two types of nodes users and movies, therefore, we are dealing with a heterogenous graph and more specifically with a bipartite graph. To create the graph we used Pytorch Geometric (PYG). On a fundamental level PYG graphs are always directed, therefore, we converted the initial graph to an undirected one since our edges do not have any particular direction and indicate a two way relationship between the nodes. This is done by creating a reverse connection for each edge originally added to the graph (this can be seen on the below outline as 'movie, rev_rating, user' connection).

<p align="center">
  <img width="230" src="https://user-images.githubusercontent.com/74935134/164025452-7f390e32-8345-41d1-93ae-505fe36751ac.png">
</p>

## Approach

### End-to-end architecture
To build an end-to-end system we used a graph encoder architecture followed by a simple MLP predictor. The input to the model are the graph and feature vectors of users and movies. First, the encoder computes the hidden representation of a user and a movie. Then, we concatenate the representations and feed them to the predictor which outputs a single number representing the predicted rating. Therefore, we consider the recommendation problem as regression, rather than classification problem

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/74935134/164025710-5b3ae7b4-ca65-4d5d-ba5f-559a67c92377.png">
</p>
<p align = "center">
  <i>Source:</i> https://arxiv.org/abs/1706.02263
</p> 

### Encoder
The purpose of the encoder is to find meaningful representations of the nodes present in the network. To do that we use graph convolutions, which is the fundamental notion underpinning most GNN systems. In a bipartite graph there are two types of nodes and therefore, at least two types of connections. In our case it's either 'user-rating-movie' or 'movie-reverse\_reating-user'. Even if we disregard the feature vector size difference, which can be overcome by adding a linear layer at the beginning, we still can't use the same parameters to process the two types of connections. Therefore, **in bipartite graphs, each type of connection has it's own set of parameters**. Moreover, the edge weights had to be included in the convolution. This is usually done by allowing edge weights to impact the message aggregation process. Having these two difficulties in mind, the heterogeneity of the graph and presence of edge weights, we decided to use [Graph Attention Networks](https://arxiv.org/abs/1710.10903) for our encoder. This state-of-the-art solution not only allows us to tackle the above issues, but also has been proven to work very well in many other areas of research.

### Predictor
Once the hidden representations of users and movies have been computed we concatenate them and input to the predictor. In our case the predictor is a Neural Network with 3 fully connected layers, where the first two use ReLU activation functions. We have made this design choice as we saw other approaches, such as using cosine similarity between the user and movie representations, give similar results. The advantage of using NNs is that their implementation was fully compatible with our PyTorch Geometric graph.

### Summary
The final model can be summarised in the outline below:

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74935134/164028410-6d8f6f8f-dd45-46aa-9c3b-4934b8171f6a.png">
</p>


## Results
The below table shows results for 3 benchmark models and our end-to-end GNN approach. The first benchmark outputs random predictions. Second benchmark predicts the average rating for a movie for all users. In this approach all users get the same recommendations, the first one being the overall highest rated movie in the dataset. The third benchmark is a standard collaborative filtering approach. The results are summarised in the below table.

  
|        **Method**       | **MAE** | **RMSE** |
|:-----------------------:|:-------:|:--------:|
| Random predictions      | 1.50    | 1.86     |
| Average rating          | 0.70    | 0.91     |
| Collaborative filtering | 0.68    | 0.88     |
| **End-to-end GNN**      | **0.99**| **1.09** |


Collaborative filtering method worked best for our dataset. This is not surprising since collaborative filtering has proven to give good results in an array of research and business applications. On the other hand the `end-to-end graph-based approach produced mixed results`. Having seen this method work in other use cases we had to reassess the assumptions made and identified three possible explanations.

First, we might be lacking training time. Given limited resources of our machines we were not able to train the model for more than 40-50 epochs. Second, we might have not fed the model with sufficient enough amount of data. Many graph-based approaches put an emphasis on how to deal with massive amounts of data, because that is what often happens in the real use cases. However, mainly for computational reasons we used a subset of the data. It could have been the case that the model was unable to learn meaningful representations due to too few examples. Lastly, we could be facing a design issue. Perhaps the model we are using is not fit for purpose, is lacking layers, width or other components. We observed a partial occurance of high bias meaning we perhaps should have increased the size of the model. This again was unfeasible due to processing power constraints of our machines.

## Credits
Nevina Dalal  
Enrico Burigana  
Thomas Gak-Deluen  
Jedrzej (Jen) Alchimowicz
