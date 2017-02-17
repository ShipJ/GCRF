import pandas as pd
import numpy as np

if __name__ == "__main__":

    path = '/Users/JackShipway/Desktop/UCLProject/Project1-Health/Data/Senegal/CDR/Temporal'

    adj_matrix = np.genfromtxt('adj_matrix.csv', delimiter=",")
    total_activity = pd.DataFrame(pd.read_csv("total_activity.csv")).as_matrix()

    q_matrix = np.array(adj_matrix / total_activity[:, 1, None])

    np.savetxt("q_matrix.csv", q_matrix, delimiter=',')
