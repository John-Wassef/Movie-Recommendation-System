# Movie Recommendation System

 This repository contains the implementation of a Movie Recommendation System using collaborative filtering and matrix factorization techniques. The system is built with PyTorch and uses the MovieLens dataset for training and evaluation.

## Overview

 The Movie Recommendation System provides personalized movie recommendations based on user ratings. It leverages matrix factorization to predict user ratings for movies they have not yet rated, using collaborative filtering to identify patterns in user behavior.

## Dataset

 The system uses the MovieLens dataset, which contains millions of movie ratings collected from the MovieLens website. For this project, the small dataset (ml-latest-small.zip) is used, which contains 100,000 ratings from 1,000 users on 9,000 movies.

## Installation
    git clone https://github.com/John-Wassef/Movie-Recommendation-System.git


## Usage

# 1. Download and extract the dataset:
    !curl http://files.grouplens.org/datasets/movielens/ml-latest-small.zip -o ml-latest-small.zip
    import zipfile
    with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

# 2. Load the data and preprocess:
    import pandas as pd

    movies_df = pd.read_csv('data/ml-latest-small/movies.csv')
    ratings_df = pd.read_csv('data/ml-latest-small/ratings.csv')

    print('The dimensions of movies dataframe are:', movies_df.shape, '\nThe dimensions of ratings dataframe are:', ratings_df.shape)
    movies_df.head()

# 3. Define the data loader class:
    from torch.utils.data.dataset import Dataset
    from torch.utils.data import DataLoader
    import torch
    import pandas as pd

    class Loader(Dataset):
        def __init__(self, ratings: pd.DataFrame):
            self.ratings = ratings

            unique_user_ids = ratings.userId.unique()
            unique_movie_ids = ratings.movieId.unique()

            self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
            self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}

            self.index_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()}
            self.index_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_index.items()}

            self.ratings['movieId'] = ratings['movieId'].map(self.movie_id_to_index)
            self.ratings['userId'] = ratings['userId'].map(self.user_id_to_index)

            self.features = self.ratings.drop(['rating', 'timestamp'], axis=1).values
            self.targets = self.ratings['rating'].values

            self.features_tensor = torch.tensor(self.features, dtype=torch.long)
            self.targets_tensor = torch.tensor(self.targets, dtype=torch.float32)

        def __getitem__(self, index: int):
            return (self.features_tensor[index], self.targets_tensor[index])

        def __len__(self) -> int:
            return len(self.ratings)

# 4. Define and train the model:
    import torch
    import numpy as np
    from torch.autograd import Variable
    from tqdm.notebook import tqdm

    class MatrixFactorization(torch.nn.Module):
        def __init__(self, n_users, n_items, n_factors=20):
            super().__init__()
            self.user_factors = torch.nn.Embedding(n_users, n_factors)
            self.item_factors = torch.nn.Embedding(n_items, n_factors)
            self.user_factors.weight.data.uniform_(0, 0.05)
            self.item_factors.weight.data.uniform_(0, 0.05)

        def forward(self, data):
            users, items = data[:, 0], data[:, 1]
            return (self.user_factors(users) * self.item_factors(items)).sum(1)

        def predict(self, user, item):
            return self.forward(user, item)

    num_epochs = 128
    cuda = torch.cuda.is_available()

    print("Is running on GPU:", cuda)

    model = MatrixFactorization(n_users, n_items, n_factors=8)
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    if cuda:
        model = model.cuda()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_set = Loader(ratings_df)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            data, target = batch
            if cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

# 5. Perform KMeans clustering on the movie embeddings:
    trained_movie_embeddings = model.item_factors.weight.data.cpu().numpy()

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10, random_state=0).fit(trained_movie_embeddings)

    for cluster in range(10):
        print("Cluster #{}".format(cluster))
        movs = []
        for movidx in np.where(kmeans.labels_ == cluster)[0]:
            movid = train_set.index_to_movie_id[movidx]
            rat_count = ratings_df.loc[ratings_df['movieId'] == movid].count()[0]
            movs.append((movie_names[movid], rat_count))
        for mov in sorted(movs, key=lambda tup: tup[1], reverse=True)[:10]:
            print("    ", mov[0])
