import pandas as pd
import numpy as np
from src.config import config


country = config.get_country()
cdr = config.get_master_cdr(country)

print cdr