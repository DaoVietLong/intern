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
    words_path        = str(row['word'])
    print("save_table_bbox->docid=", docid, ", words_path=", words_path)
    #if words_path == "" or words_path is None:
    #    print("save_table_bbox->done!!!")
    #    return ""
    img = os.path.join(args.image_dir, docid + "-table.jpg")
    #words_path = os.path.join(args.words_dir, docid + "_words.json")
    topath = os.path.join(args.output_dir, docid + "-token.jpg")
    
    data = ds.metadata_load_json(words_path)
    #print("save_table_bbox->tokens.data =", data)
    tokens = data['words']
    print("save_table_bbox->tokens = ", len(tokens))
    #if tokens is None:        
    #    return ""
    if len(tokens) < 1:        
        return ""
    
    img = Image.open(img)
    if img.mode != 'RGB':  
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img) 
    for token in tokens:
        bbox = token["bbox"]
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        shape = [start_point, end_point]
        draw.rectangle(shape, outline ="red") 
    img.save(topath)
def main(args):
    print("****** START CHECK ******\n\n")
    print ("args")
    print (args)
    
    data_type = "table-words"
    datas = ds.metadata_load(args.data_dir, data_type)
    print("datas")
    print(datas)
    ds.ensure_dir(args.output_dir)
    for ind, row in datas.iterrows():
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
        '--image_dir',
        type=str,
        default='data/final-crop',
        help=''
    )
    parser.add_argument(
        '--words_dir',
        type=str,
        default='data/final-words',
        help=''
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/check-tokens',
        help=''
    )
    FLAGS = parser.parse_args()
    print("****** PARAMS ******\n\n")
    print (FLAGS)  
    main(FLAGS)
    