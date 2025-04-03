import argparse
import os
import numpy as np
import pandas as pd
import numpy as np
from utils import dataset as ds
from utils import visual as viz
import functools
print = functools.partial(print, flush=True)
 
 
def main(args):
    print("****** START PREPARE ANNOTATION ******\n\n")
    print ("args")
    print (args)
    
    data_prefix = "yooz_"
    suffix      = "full"
    datas = ds.yooz_meta_data_load(args.data_json, args.data_img)
    
    print ("datas")
    print (datas)
    ds.ensure_dir(args.output_dir)
    ds.ensure_dir(args.output_ground_truth)
    ds.metadata_save(args.output_dir, datas, "table")
    
    '''    
    for ind, row in datas.iterrows():
        output_name = os.path.join(args.output_ground_truth, row['id'] + "_gt.jpg")
        #viz.visualize_doc(row['img'], row['json'], output_name)
    '''
    table_bbox_datas = ds.load_table_bbox(args.data_table_bbox)
    print("table_bbox_datas")
    print(table_bbox_datas)
    
    for ind, row in datas.iterrows():
        docid = row['id']
        table_bbox = ds.find_doc(docid, table_bbox_datas)
        print("docid=", docid)
        print(table_bbox)
        datas.at[ind,"table_x"] = table_bbox["x"]
        datas.at[ind,"table_y"] = table_bbox["y"]
        datas.at[ind,"table_width"] = table_bbox["width"]
        datas.at[ind,"table_height"] = table_bbox["height"]
        #break
    
    ds.metadata_save(args.output_dir, datas[datas["table_width"] > 0], "table-bbox-OK")
    ds.metadata_save(args.output_dir, datas[datas["table_width"] == 0], "table-bbox-NG")
    
    print("****** END PREPARE ANNOTATION ******\n\n")
    return 1
if __name__ == "__main__":
    
    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_json',
        type=str,
        default='data/Yooz/json/Bergstrom',
        help=''
    )
    parser.add_argument(
        '--data_img',
        type=str,
        default='data/Yooz/test/img_xml',
        help=''
    )
    parser.add_argument(
        '--data_table_bbox',
        type=str,
        default='data/Yooz/table-bbox.json',
        help=''
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/final',
        help=''
    )
    parser.add_argument(
        '--output_ground_truth',
        type=str,
        default='data/ground_truth',
        help=''
    )
    FLAGS = parser.parse_args()
    print("****** PARAMS ******\n\n")
    print (FLAGS)  
    main(FLAGS)
    