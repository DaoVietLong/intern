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
 

def save_table_bbox(row, args):
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
    #return 1
    #img = Image.open(img)
    #if img.mode != 'RGB':  
    #    img = img.convert('RGB')
    #start_point = (table_x, table_y)
    #end_point = (table_x + table_width, table_y + table_height)
    
    start_point = (td[0], td[1])
    end_point = (td[2], td[3])
    
    color = (255, 0, 0)
    thickness = 2
    
    topath = os.path.join(args.output_dir, docid + "-table.jpg")
    
    shape = [start_point, end_point]
    img = Image.open(img)
    if img.mode != 'RGB':  
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img)   
    draw.rectangle(shape, outline ="red") 
    img.save(topath)
def main(args):
    print("****** START CHECK ******\n\n")
    print ("args")
    print (args)
    
    data_type = "table-bbox-OK"
    datas = ds.metadata_load(args.data_dir, data_type)
    checks = ["1_b7816646-d903-4165-b0e7-d39485c49000-17125-19558-1704471085695-yooz-1-5-24_1"]
    print("datas")
    print(datas)
    ds.ensure_dir(args.output_dir)
    for ind, row in datas.iterrows():
        if len(checks) > 0:
            docid = row['id']
            if docid in checks:
                save_table_bbox(row, args)
        else:
            save_table_bbox(row, args)
        #break
    print("****** END CHECK ******\n\n")
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
        '--output_dir',
        type=str,
        default='data/ground_truth_table_bbox',
        help=''
    )
    FLAGS = parser.parse_args()
    print("****** PARAMS ******\n\n")
    print (FLAGS)  
    main(FLAGS)
    