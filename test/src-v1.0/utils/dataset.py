"""
Simple Framework for Incremental Learning

Authors: PHAM Tri Cong*+
Affiliation: *L3i, La Rochelle University, La Rochelle, 17000, France.
             +Yooz, 1 Rue Fleming, La Rochelle, 17000, France.
Corresponding Email: cong.pham@univ-lr.fr
Date: Oct 20, 2022

"""

import argparse
import os
import numpy as np
import pandas as pd
import re
import pickle
import csv
from PIL import Image
import json
'''
common function
'''
def ensure_dir_for_file(file_path):
    directory = os.path.dirname(file_path)
    ensure_dir(directory)
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def put_content_to_file(data, path):
    with open(path, 'wb') as ofile:
        pickle.dump(data, ofile)
def get_content_from_file(path):
    with open(path, 'rb') as data:
        return pickle.load(data)
def metadata_load (data_dir, dtype, encoding='latin-1'):
    sep=","
    path = os.path.join(data_dir, dtype + ".csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep=sep, encoding = encoding)
def metadata_save(data_dir, df, dtype = "", encoding='latin-1'):
    path = os.path.join(data_dir, dtype + ".csv")
    df.to_csv(path, encoding = encoding, index=False)
def txt_save(path, txt, encoding='latin-1'):
    txt = txt.encode("utf-8").decode(encoding)
    f = open(path, "w", encoding=encoding)
    f.write(txt)
    f.close()  
def text_csv_load (data_dir, dtype, sep = ",", names=None, encoding='latin-1'):
    #sep=","
    file_path = os.path.join(data_dir, dtype)
    if names is None:
        return pd.read_csv(file_path, sep = sep, encoding = encoding)
    return pd.read_csv(file_path, sep = sep, encoding = encoding, names = names)
def text_csv_save(data_dir, datas, type = "", encoding='latin-1'):
    datas["text"] = datas["path"].map(lambda x: "" if (not x) else text_data_load(x))
    text_csv_save_with_text(data_dir, datas, type, encoding)
    
def metadata_load_json(path, encoding='latin-1', default = []):
    if not os.path.exists(path):
        return default
    f = open(path,encoding = encoding)
    data = json.load(f)
    return data
def metadata_save_json(data, path, encoding='latin-1'):
    f = open(path, "w", encoding = encoding)
    json.dump(data, f)   
def table_bbox(item, l):
    ret = []
    print("table_bbox->item")
    print(item)
    print("table_bbox->annotations")
    print(item["annotations"])
    for annotation in item["annotations"]:
        for data in annotation["result"]:
            val = data["value"]
            print("table_bbox->val")
            print(val) 
            if l not in val["rectanglelabels"]:
                continue
            if val["x"] >= 100.0:
                continue
            if val["y"] >= 100.0:
                continue
            if val["width"] >= 100.0:
                continue
            if val["height"] >= 100.0:
                continue
            ret = val
            ret["original_width"]   = data["original_width"]
            ret["original_height"]  = data["original_height"]
            ret["file_upload"]      = item["file_upload"]
            
            ret["x"]        = ret["original_width"] * ret["x"]  / 100.0
            ret["y"]        = ret["original_height"] * ret["y"]  / 100.0
            ret["width"]    = ret["original_width"] * ret["width"]  / 100.0
            ret["height"]   = ret["original_height"] * ret["height"]  / 100.0
            
    return ret
def find_doc(docid, maps):
    ret = {"x" : 0, "y" : 0, "width" : 0, "height" : 0}
    smax = 0
    for k in maps.keys():
        c = find_max_matching(docid, k)
        if c > smax:
            ret = maps[k]
            smax = c
    return ret
def find_max_matching(ori, sub):
    on = len(ori)
    sn = len(sub)
    if on < sn:
        return 0
    if on == sn:
        if ori == sub: 
            return on
        return 0
    #on > sn    
    for i in range(sn):
        if ori[i] != sub[i]:
            return i
    return sn
def label_to_key(docid):
    return docid.split("-", 1)[1]
def load_table_bbox(path, encoding='latin-1'):
    ret = {}
    bbox_datas = metadata_load_json(path)
    for data in bbox_datas:
        bbox = table_bbox(data, "Table")
        docid = yooz_convert_to_id(bbox["file_upload"])
        docid = yooz_convert_to_id_bbox(docid)
        ret[docid] = bbox
    
    return ret
def yooz_convert_to_id_bbox(x):
    x = x.replace("_00001", "")
    x = x.replace("{", "")
    x = x.replace("}", "")
    return label_to_key(x)
    return x
def yooz_convert_to_id(x):
    return os.path.splitext(x.lower())[0]
def folder_scan_by_ext(datas, data_dir, ext = ".tif"):
    obj = os.scandir(data_dir)
    for entry in obj :
        if entry.is_file():
            fname = entry.name.lower()
            if not fname.endswith(ext):
                continue
            fileid = yooz_convert_to_id(entry.name)
            path = os.path.join(data_dir, entry.name)
            datas[fileid] = path
    return datas

   
def yooz_meta_data_load (data_json, data_img):
    fmap = {}
    folder_scan_by_ext(fmap, data_json, ".json")
    ret = []
    notfound = []
    for docid in fmap.keys():
        json_path = fmap[docid]
        item = {"id": docid, "json": path_format(json_path)}
        img_name = docid + "_00001.jpg"        
        img_path = path_format(os.path.join(data_img, img_name))
        if os.path.exists(img_path):
            item["img"] = img_path            
        else:
            img_name = docid + ".jpg"        
            img_path = path_format(os.path.join(data_img, img_name))
            if os.path.exists(img_path):
                item["img"] = img_path                
            else:
                continue
        ret.append(item) 
    ret = pd.DataFrame(ret)    
    return ret
def path_format (p):
    return p.replace("\\", "/")
def put_content_to_file(data, path):
    with open(path, 'wb') as ofile:
        pickle.dump(data, ofile)
def get_content_from_file(path):
    with open(path, 'rb') as data:
        return pickle.load(data)    
