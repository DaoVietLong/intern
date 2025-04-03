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
import xml.etree.ElementTree as ET
from fitz import Rect
print = functools.partial(print, flush=True)

def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)
    
    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area
    
    return 0
def ocr_to_tokens(tree, scale):
    ret = []
    lst = tree.findall('block/line')
    word_num = 1
    for line in lst:
        lst = line.findall('word')
        for word in lst:
            token = {}
            token['flags'] = 0
            #token['span_num'] = word_num
            #token['line_num'] = 0
            #token['block_num'] = 0
            txt = word.get('value')
            if txt != "":
                #left="352" top="196" right="456" bottom="225"
                bbox = [to_x(word.attrib["left"], scale),
                        to_x(word.attrib["top"], scale),
                        to_x(word.attrib["right"], scale),
                        to_x(word.attrib["bottom"], scale)
                        ]
                token['bbox'] = bbox
                token['text'] = txt
                ret.append(token)
    return ret

def to_x(v, scale):
    return round(scale * int(v), 5)
def xml_to_tokens(xml_path, scale = 1.0):
    with open(xml_path, encoding='utf-8') as f:
        tree = ET.parse(f)
        print ("xml_to_tokens.call ocr_to_tokens")
        return ocr_to_tokens(tree, scale)
    print ("xml_to_tokens.no tokens")
    return [] 
def crop_bbox_tokens(tokens, bbox):
    ret = []
    crop_bbox = [bbox[0], bbox[1], bbox[0], bbox[1]]
    for token in tokens:
        token["bbox"][0] = token["bbox"][0] - bbox[0]
        token["bbox"][1] = token["bbox"][1] - bbox[1]
        token["bbox"][2] = token["bbox"][2] - bbox[0]
        token["bbox"][3] = token["bbox"][3] - bbox[1]
        
        if token["bbox"][0] < 0:
            token["bbox"][0] = 0
        if token["bbox"][1] < 0:
            token["bbox"][0] = 0
        if token["bbox"][2] > (bbox[2] - bbox[0]):
            token["bbox"][2] = bbox[2] - bbox[0]
        if token["bbox"][3] > (bbox[3] - bbox[1]):
            token["bbox"][3] = bbox[3] - bbox[1]
        ret.append(token)    
    return ret
    
def save_table_words(row, args):
    docid       = row['id']
    img         = row['img']
    json        = row['json']
    
    data = ds.metadata_load_json(json)
    print("save_table_bbox->json data")
    print(data)
    table_bbox = data["TD"][0][0]
    print("save_table_bbox->table_bbox")
    print(table_bbox)
    
    
    xml_path = img.replace(".jpg", ".xml")
    print("save_table_bbox->xml_path=", xml_path)
    tokens = xml_to_tokens(xml_path)
    
    if (len(table_bbox) < 4):
        table_tokens_crop = []
    else:
        table_tokens = [token for token in tokens if iob(token['bbox'], table_bbox) >= 0.5]
        print("\nsave_table_bbox->tokens=", tokens)
        print("\nsave_table_bbox->table_tokens=", table_tokens)
        table_tokens_crop = crop_bbox_tokens(table_tokens, table_bbox)
        print("\nsave_table_bbox->cropped table_tokens=", table_tokens_crop)
        print("save_table_bbox->tokens len=", len(tokens), ", table_tokens len=", len(table_tokens))
    final_data = {"words":table_tokens_crop}
    file_name = os.path.basename(img)
    topath = os.path.join(args.output_dir, file_name.replace(".jpg", "_words.json"))
    
    #print("save_table_words->topath=", topath)
    ds.metadata_save_json(final_data, topath)
    return ds.path_format(topath)
def main(args):
    print("****** START CHECK ******\n\n")
    print ("args")
    print (args)
    
    #data_type = "table-bbox-OK"
    data_type = "table-crop"
    datas = ds.metadata_load(args.data_dir, data_type)
    print("datas")
    print(datas)
    ds.ensure_dir(args.output_dir)
    table_path = ""
    for ind, row in datas.iterrows():
        table_path = save_table_words(row, args)
        datas.at[ind,"word"] = table_path
        #break
    ds.metadata_save(args.data_dir, datas, "table-words")  
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
        default='data/final-words',
        help=''
    )
    FLAGS = parser.parse_args()
    print("****** PARAMS ******\n\n")
    print (FLAGS)  
    main(FLAGS)
    