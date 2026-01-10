#!/usr/bin/env python3

import mlcroissant as mc
from mlcroissant import Dataset
import warnings
import logging

# This silences the "root" logger warnings specifically --> een random warning bij Preview of records
logging.getLogger().setLevel(logging.ERROR) 
# This silences general python warnings (beetje risky maar you never know)
#warnings.filterwarnings('ignore')

ds = Dataset(jsonld="https://huggingface.co/api/datasets/patriziobellan/PETv11/croissant")
records = ds.records("relations-extraction")

 # Extract records, verander i als je er meer wil zien (of minder)
print("Preview of records:\n")
for i, record in enumerate(records):
    print(f"Record {i+1}:")
    for key, value in record.items():
        print(f"  {key}: {value}")
    print()
    if i >= 4:
        break

#print("Files identified by Croissant:")
#for file in ds.metadata.distribution:
#    print(f" - {file.name}: {file.content_url}")