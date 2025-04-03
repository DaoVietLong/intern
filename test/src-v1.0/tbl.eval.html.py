import argparse
import os
import numpy as np
import pandas as pd
import numpy as np
from utils import dataset as ds
from utils import visual as viz
import functools
import cv2
import xml.etree.ElementTree as ET
from PIL import Image, ImageFont, ImageDraw
from fitz import Rect
import postprocess 
from collections import OrderedDict, defaultdict
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
def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table spanning cell': 3,
            'table projected row header': 4,
            'table column header': 5,
            'no object': 6 
        }
    elif data_type == 'detection':
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map 
detection_class_thresholds = {
    "table": 0.5,
    "table rotated": 0.5,
    "no object": 10
}

structure_class_thresholds = {
    "table": 0.5,
    "table column": 0.5,
    "table row": 0.5,
    "table column header": 0.5,
    "table projected row header": 0.5,
    "table spanning cell": 0.5,
    "no object": 10
}
def load_result_bbox(row, args):
    docid       = row['id']
    path = os.path.join(args.result_dir, docid + "-table_0_objects.json")
    print("load_result_bbox->docid=", docid, ", path=", path)
    data = ds.metadata_load_json(path, default = [])
    return data
def save_table_bbox(row, args):
    docid       = row['id']
    words_path        = str(row['word'])
    print("save_table_bbox->docid=", docid, ", words_path=", words_path)
    #if words_path == "" or words_path is None:
    #    print("save_table_bbox->done!!!")
    #    return ""
    img = os.path.join(args.image_dir, docid + "-table.jpg")
    #words_path = os.path.join(args.words_dir, docid + "_words.json")
    topath = os.path.join(args.output_dir, docid + "-result-token.jpg")
    
    file_name = os.path.basename(row['img'])
    #topath_html = os.path.join(args.output_dir_html, file_name.replace(".jpg", ".html"))
    topath_html_w_txt = os.path.join(args.output_dir_html_w_txt, docid + ".html")
    topath_html_wo_txt = os.path.join(args.output_dir_html_wo_txt, docid + ".html")
    topath_csv = os.path.join(args.output_dir, file_name.replace(".jpg", ".csv"))
    
    
    data = ds.metadata_load_json(words_path)
    #print("save_table_bbox->tokens.data =", data)
    tokens = data['words']
    print("save_table_bbox->tokens = ", len(tokens))
    #if tokens is None:        
    #    return ""
    if len(tokens) < 1:        
        return ""
    
    #print("save_table_bbox->tokens = ", tokens)
    result_tsr = load_result_bbox(row, args)
    #print("save_table_bbox->result_tsr = ", result_tsr)
    print("save_table_bbox->tokens = ", len(tokens))
    print("save_table_bbox->result_tsr = ", len(result_tsr))
    if len(result_tsr) < 1:        
        return ""
    tokens_in = []    
    for idx, item in enumerate(result_tsr):
        #print("item bbox:", item['bbox'])
        cell_txts = []
        txt_inds = []
        for tidx, token in enumerate(tokens):
            #print("text bbox:", token['bbox'])
            ch = iob(token['bbox'], item['bbox'])
            #print("save_table_bbox->iob = ", ch)
            if ch >= 0.5:
                cell_txts.append(token['text'])
                txt_inds.append(tidx)
                tokens_in.append(tidx)
        #print("cell_txts=", cell_txts)
        item["cell text"] = ' '.join(cell_txts) 
        item["tokens index"] = txt_inds
        
        #tokens_in.append(txt_inds)
    print("save_table_bbox->result_tsr = ", result_tsr)
    #return ""
    img = Image.open(img)
    if img.mode != 'RGB':  
        img = img.convert('RGB')
    draw = ImageDraw.Draw(img) 
    
    width = img.width 
    height = img.height 
    #for token in tokens:
    for idx in tokens_in:
        token = tokens[idx]
        bbox = token["bbox"]
        if bbox[3] > height:
            bbox[3] = height
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        shape = [start_point, end_point]
        draw.rectangle(shape, outline ="red") 
    font = ImageFont.truetype(r'Arial.ttf', 20) 
    for item in result_tsr:
        bbox = item["bbox"]        
        
        if bbox[3] > height:
            bbox[3] = height
        if bbox[0] > bbox[2]:
            print("save_table_bbox->ERROR---bbox = ", bbox)
            continue
        
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        shape = [start_point, end_point]
        draw.rectangle(shape, outline ="blue") 
        #draw.text((5, 5), text, font = font, align ="left")  
        draw.text(start_point, item["cell text"], font = font, align ="left", fill ="red")  
    img.save(topath)
    
    table_html = cells_to_html(result_tsr, True)
    ds.txt_save(topath_html_w_txt, table_html)
    
    table_html_wo_txt = cells_to_html(result_tsr, False)
    ds.txt_save(topath_html_wo_txt, table_html_wo_txt)
    #f = open(topath_html, "w", encoding='latin-1')
    #f.write(table_html)
    #f.close()
    
    table_csv = cells_to_csv(result_tsr)
    ds.txt_save(topath_csv, table_csv)
    #f = open(topath_csv, "w", encoding='latin-1')
    #f.write(table_csv)
    #f.close()
    
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
def cells_to_csv(cells):
    if len(cells) > 0:
        num_columns = max([max(cell['column_nums']) for cell in cells]) + 1
        num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
    else:
        return

    header_cells = [cell for cell in cells if cell['column header']]
    if len(header_cells) > 0:
        max_header_row = max([max(cell['row_nums']) for cell in header_cells])
    else:
        max_header_row = -1

    table_array = np.empty([num_rows, num_columns], dtype="object")
    if len(cells) > 0:
        for cell in cells:
            for row_num in cell['row_nums']:
                for column_num in cell['column_nums']:
                    table_array[row_num, column_num] = cell["cell text"]

    header = table_array[:max_header_row+1,:]
    flattened_header = []
    for col in header.transpose():
        flattened_header.append(' | '.join(OrderedDict.fromkeys(col)))
    df = pd.DataFrame(table_array[max_header_row+1:,:], index=None, columns=flattened_header)

    return df.to_csv(index=None)
def main(args):
    print("****** START CHECK ******\n\n")
    print ("args")
    print (args)
    
    data_type = "table-words"
    datas = ds.metadata_load(args.data_dir, data_type)
    print("datas")
    print(datas)
    ds.ensure_dir(args.output_dir)
    ds.ensure_dir(args.output_dir_html_w_txt)
    ds.ensure_dir(args.output_dir_html_wo_txt)
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
        default='data/final-tables',
        help=''
    )
    parser.add_argument(
        '--words_dir',
        type=str,
        default='data/final-words',
        help=''
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default='results/final-tables-FinTabNet-TSR',
        help=''
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/evaluation/FinTabNet-TSR',
        help=''
    )
    parser.add_argument(
        '--output_dir_html_w_txt',
        type=str,
        default='results/html/FinTabNet-TSR-w-Text',
        help=''
    )
    parser.add_argument(
        '--output_dir_html_wo_txt',
        type=str,
        default='results/html/FinTabNet-TSR-wo-Text',
        help=''
    )
    FLAGS = parser.parse_args()
    print("****** PARAMS ******\n\n")
    print (FLAGS)  
    main(FLAGS)
    