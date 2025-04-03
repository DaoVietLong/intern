import argparse
import os
import numpy as np
import pandas as pd
import numpy as np
from utils import dataset as ds
from utils import visual as viz
import functools
import cv2
from PIL import Image, ImageDraw 
print = functools.partial(print, flush=True)
 
def get_table_small_bbox(data):
    td = data["TD"][0][0]
    table_cells = data["TSR"][0][0]["data"]
    print("save_table_bbox->table_cells")
    print(table_cells)
    max_y = 0
    min_h = -1
    #min_h = 30
    
    for idx, item in enumerate(table_cells):
        
        if item["bbox"][3] > max_y:
            max_y = item["bbox"][3]
        h = item["bbox"][3] - item["bbox"][1]
        if (h < min_h) or (min_h == -1):
            min_h = h
    max_y = max_y + min_h/3
    if max_y < td[3]:
        td[3] = max_y
    return td
def save_table(row, args):
    
    docid       = row['id']
    img         = row['img']
    json        = row['json']
    table_x     = row['table_x']
    table_y     = row['table_y']
    table_width = row['table_width']
    table_height= row['table_height']
    data = ds.metadata_load_json(json)
    print("save_table_bbox->json data")
    print(data)
    
    
    td = data["TD"][0][0]
    print("save_table_bbox->td")
    print(td)
    if len(td) < 4:
        return ""
    
    if args.type == "small":
        td = get_table_small_bbox(data)
        print("save_table_bbox->td.small_bbox")
        print(td)
    #return 1
    start_point = (td[0], td[1])
    end_point = (td[2], td[3])
    
    topath = os.path.join(args.output_dir, docid + "-table.jpg")
    
    img = Image.open(img)
    if img.mode != 'RGB':  
        img = img.convert('RGB')
    img = img.crop(td)
    img.save(topath)
    return ds.path_format(topath)
def save_crop(datas, args):
    output_table = "table-crop"
    suff = ""
    if args.type == "small":
        suff = "-small"
        args.output_dir = args.output_dir + suff
        output_table = output_table + suff
    print("datas")
    print(datas)
    
    ds.ensure_dir(args.output_dir)
    for ind, row in datas.iterrows():
        table_path = save_table(row, args)
        datas.at[ind,"crop"] = table_path
        #break
    ds.metadata_save(args.data_dir, datas, output_table) 
def main(args):
    print("****** START CROP ******\n\n")
    print ("args")
    print (args)
    
    data_type = "table-bbox-OK"
    datas = ds.metadata_load(args.data_dir, data_type)
    return save_crop(datas, args)
    output_table = "table-crop"
    suff = ""
    if args.type == "small":
        suff = "-small"
        args.output_dir = args.output_dir + suff
        output_table = output_table + suff
    print("datas")
    print(datas)
    
    ds.ensure_dir(args.output_dir)
    for ind, row in datas.iterrows():
        table_path = save_table(row, args)
        datas.at[ind,"crop"] = table_path
        break
    ds.metadata_save(args.data_dir, datas, output_table)    
    print("****** END CROP ******\n\n")
    return 1    
if __name__ == "__main__":
    
    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/final',
        help=''
    )    
    parser.add_argument(
        '--type',
        type=str,
        default='small',
        help=''
    )    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/final-tables',
        help=''
    )
    FLAGS = parser.parse_args()
    print("****** PARAMS ******\n\n")
    print (FLAGS)  
    main(FLAGS)
    