# combinations to test

episodes: 150
batch_size: 16
epsilon_decay: 0.9
epsilon_min: 0.2

episodes: 150
batch_size: 16
epsilon_decay: 0.99
epsilon_min: 0.1

episodes: 200
batch_size: 32
epsilon_decay: 0.9
epsilon_min: 0.4

episodes: 250
batch_size: 16
epsilon_decay: 0.95
epsilon_min: 0.2

episodes: 300
batch_size: 16
epsilon_decay: 0.9
epsilon_min: 0.3

episodes: 500
batch_size: 16
epsilon_decay: 0.99
epsilon_min:  0.1

episodes: 1000
batch_size: 32
epsilon_decay: 0.995
epsilon_min: 0.2

parametergrid:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html

beispiel usage:
https://stackoverflow.com/questions/74493645/can-you-iterate-over-hyperparameters-in-scikit