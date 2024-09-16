from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample data
data = np.array(['A', 'B', 'C', 'A', 'B', 'C']).reshape(-1, 1)

# Create the encoder and fit it
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(data)

print(encoded_data)

[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]



 # write one hot encoding from scratch
categories = np.array(['A', 'B', 'C', 'A', 'B', 'C']).tolist()

category_to_index = {category : index for index, category in enumerate(sorted(set(categories)))}

encoded_data = []

for category in categories:
    encoded_curr = [0] * len(category_to_index)
    encoded_curr[category_to_index[category]] = 1
    encoded_data.append(encoded_curr)

print(encoded_data)
# [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]