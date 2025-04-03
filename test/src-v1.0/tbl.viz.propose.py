import argparse
import os
from utils import dataset as ds
import functools
print = functools.partial(print, flush=True)
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw 
import shutil
from utils import visual as viz
def viz_doc(docid, args):
    img_path = os.path.join(args.image_dir, docid + "-table.jpg")
    
    result_path = os.path.join(args.result_dir, docid + "-table_0_objects.json")
    roi_path = os.path.join(args.result_cache_dir, docid + "-table_roi.npy")
    output_path = os.path.join(args.output_dir, docid + "-roi.jpg")
    
    print("viz_doc->roi_path=", roi_path)
    
    output_dict = ds.get_content_from_file(roi_path)
    
    print("viz_doc->output_dict")
    print(output_dict)
    #return 1
    
    img = Image.open(img_path)
    if img.mode != 'RGB':  
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img) 
    
    #for bbox in output_dict["pred_boxes"]:
    for idx, bbox in enumerate(output_dict["pred_boxes"]):
        score = output_dict["scores"][idx]
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        shape = [start_point, end_point]
        color = "green"
        if score < 0.5:
            color = "red"
        draw.rectangle(shape, outline = color) 
    img.save(output_path)
    
    return 1
def copy_data(docid, args):
    img_path = os.path.join(args.image_dir, docid + "-table.jpg")    
    predicted_path = os.path.join(args.result_dir, docid + "-table_fig_cells.jpg")
    destination = os.path.join(args.output_dir, docid + "-table_fig_cells.jpg")
    shutil.copyfile(predicted_path, destination)
    
    json_path = os.path.join(args.data_json, docid + ".json")
    output_gt = os.path.join(args.output_dir, docid + "-table-gt.jpg")
    viz.visualize_cropped_table(img_path, json_path, output_gt)
        
def main(args):
    print("****** START EVALUATION ******\n\n")
    print ("args")
    print (args)
    docids = []
    #docids.append("3_b7816646-d903-4165-b0e7-d39485c49000-17125-25786-1707146845976-payables-02-05-24_3")
    #docids.append("1_1704304244596-1004503432_1")
    
    datas = ds.metadata_load (args.csv_dir, args.result_file + "-S10")
    
    output_dir = args.output_dir
    #ds.ensure_dir(output_dir)
    #for docid in docids:
    for ind, row in datas.iterrows():
        docid = row["id"]
        
        args.output_dir = os.path.join(output_dir, row["bins"])
        ds.ensure_dir(args.output_dir)
        viz_doc(docid, args)
        copy_data(docid, args)
        #break
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
        '--csv_dir',
        type=str,
        default='results/evaluation-small',
        help=''
    )
    parser.add_argument(
        '--result_file',
        type=str,
        default='teds_scores-wo-text-small',
        help=''
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='data/final-tables-small',
        help=''
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default='results/final-tables-FinTabNet-TSR-small',
        help=''
    )
    parser.add_argument(
        '--result_cache_dir',
        type=str,
        default='results/cache/final-tables-FinTabNet-TSR-small',
        help=''
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/viz-roi',
        help=''
    )
    
    FLAGS = parser.parse_args()
    print("****** PARAMS ******\n\n")
    print (FLAGS)  
    main(FLAGS)
    