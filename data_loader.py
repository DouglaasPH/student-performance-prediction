import os
import shutil

import pandas as pd
import kagglehub

def load_data():
    # Download dataset (cache padrão)
    path = kagglehub.dataset_download("aljarah/xAPI-Edu-Data")
    print("Cache path:", path)

    # Arquivo original dentro do cache
    source_file = os.path.join(path, "xAPI-Edu-Data.csv")

    # Pasta local desejada
    local_dir = "data"
    os.makedirs(local_dir, exist_ok=True)

    # Novo caminho
    target_file = os.path.join(local_dir, "xAPI-Edu-Data.csv")

    # Copiar arquivo para pasta local
    shutil.copy(source_file, target_file)

    # Ler dataset da pasta local
    df = pd.read_csv(target_file)

    return df
