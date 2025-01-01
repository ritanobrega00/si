import sys
import os

# Adiciona o diretório que contém a pasta datasets ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Agora você pode importar de datasets
from datasets import DATASETS_PATH
