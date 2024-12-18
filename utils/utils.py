import numpy as np
from datetime import datetime

def convert_to_float(d):
    d = d.to_dict()
    new_data = {}
    for k, v in d.items():
        new_data[k] = v
        if isinstance(v, np.generic):
            new_data[k] = new_data[k].item()
        if k == "output":
            if 'retrieval_result' in new_data[k]:
                for res in new_data[k]['retrieval_result']:
                    if 'metadata' in res:
                        for kk in res['metadata']:
                            if isinstance(res['metadata'][kk], datetime):
                                res['metadata'][kk] = res['metadata'][kk].isoformat()

    return new_data