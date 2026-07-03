import os
import sys
app_local_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(app_local_path)
from clopa.app import InferApp
