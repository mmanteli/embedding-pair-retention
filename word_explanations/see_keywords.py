from glob import glob
import sys
import os
import json
import datasets

def read_results(path):
    #print(f"\tTrying to read {path}")
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
        return results
    else:
        print(f"Cannot find file {path}")
        return None

stats_path = sys.argv[1]
assert os.path.exists(stats_path)

dict_path = sys.argv[2]
assert os.path.exists(dict_path)

stats = read_results(stats_path)
j = stats["peak_indices"]
if len(j) == 0:
    print("No peak.")
    exit()
print(f"peak is at {j}")

kws = datasets.load_from_disk(dict_path)
for j_ in j:
    print(kws[str(j_)]["texts"][:10])
    print(kws[str(j_)]["texts"][-10:]) 


