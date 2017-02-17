import pandas as pd
import numpy as np
import math
import os.path

if __name__ == "__main__":

    deg_matrix = pd.DataFrame(pd.read_csv()).as_matrix()
    self_matrix = deg_matrix.diagonal()

    total_activity = pd.DataFrame(pd.read_csv("/Users/JackShipway/Desktop/UCLProject/Project1-Health/Senegal/CDR/..."))

    in_out_activity = total_activity - self_matrix
    introversion = self_matrix / in_out_activity

    # Normalise by population, same method as before

