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
def cells_to_html(cells, incl_txt = True):
    cells = sorted(cells, key=lambda k: min(k['column_nums']))
    cells = sorted(cells, key=lambda k: min(k['row_nums']))

    table = ET.Element("table")
    current_row = -1

    for cell in cells:
        this_row = min(cell['row_nums'])

        attrib = {}
        colspan = len(cell['column_nums'])
        if colspan > 1:
            attrib['colspan'] = str(colspan)
        rowspan = len(cell['row_nums'])
        if rowspan > 1:
            attrib['rowspan'] = str(rowspan)
        if this_row > current_row:
            current_row = this_row
            if cell['column header']:
                cell_tag = "th"
                row = ET.SubElement(table, "thead")
            else:
                cell_tag = "td"
                row = ET.SubElement(table, "tr")
        tcell = ET.SubElement(row, cell_tag, attrib=attrib)
        if incl_txt:
            tcell.text = cell['cell text']
        else:
            tcell.text = ""

    return str(ET.tostring(table, encoding="unicode", short_empty_elements=False))    
def save_table_gt(row, args):
    docid       = row['id']
    img         = row['img']
    json        = row['json']
    
    data = ds.metadata_load_json(json)
    print("save_table_bbox->json data")
    print(data)
    table_bbox = data["TD"][0][0]
    print("save_table_bbox->table_bbox")
    print(table_bbox)
    
    table_cells = data["TSR"][0][0]["data"]
    print("save_table_bbox->table_cells")
    print(table_cells)
    #for cell in table_cells:
    for idx, item in enumerate(table_cells):
        item["column header"] = item["column_header"]
        item["cell text"] = item["cell_text"]        
    
    table_html_txt = cells_to_html(table_cells, True)
    table_html_wo_txt = cells_to_html(table_cells, False)
    
    topath_html_txt = os.path.join(args.output_dir_w_text, docid +  ".html")
    topath_html_wo_txt = os.path.join(args.output_dir_wo_text, docid +  ".html")
    ds.txt_save(topath_html_txt, table_html_txt)
    ds.txt_save(topath_html_wo_txt, table_html_wo_txt)
    return ds.path_format(topath_html_txt) , ds.path_format(topath_html_wo_txt)
     
def main(args):
    print("****** START TEDS GROUND TRUTH ******\n\n")
    print ("args")
    print (args)    
    data_type = "table-words"    
    datas = ds.metadata_load(args.data_dir, data_type)
    print("datas")
    print(datas)
    ds.ensure_dir(args.output_dir_w_text)
    ds.ensure_dir(args.output_dir_wo_text)
    table_path = ""
    for ind, row in datas.iterrows():
        table_path_txt, table_path_wo_txt  = save_table_gt(row, args)
        datas.at[ind,"teds"] = table_path_txt
        datas.at[ind,"teds_wo"] = table_path_wo_txt
        #break
    #ds.metadata_save(args.data_dir, datas, "table-teds-gt")  
    print("****** END TEDS GROUND TRUTH ******\n\n")
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
        '--output_dir_w_text',
        type=str,
        default='data/ground-truth-w-text',
        help=''
    )
    parser.add_argument(
        '--output_dir_wo_text',
        type=str,
        default='data/ground-truth-wo-text',
        help=''
    )
    FLAGS = parser.parse_args()
    print("****** PARAMS ******\n\n")
    print (FLAGS)  
    main(FLAGS)
    