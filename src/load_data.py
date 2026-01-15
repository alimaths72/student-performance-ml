import pandas as pd
import os

# Project root (one level above src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def load_and_merge_data():
    mat_path = os.path.join(DATA_DIR, "student-mat.csv")
    por_path = os.path.join(DATA_DIR, "student-por.csv")

    # IMPORTANT: Different separators
    mat = pd.read_csv(mat_path, sep=",")     # MAT uses comma
    por = pd.read_csv(por_path, sep=";")     # POR uses semicolon

    data = pd.concat([mat, por], ignore_index=True)

    return data
