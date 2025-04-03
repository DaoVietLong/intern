import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from PIL import Image
import json
import os
import numpy as np

def visualize_cells_table(img, cells, out_path):  
    import matplotlib  
    matplotlib.use('Agg')  # Use Agg backend for better performance  
    
    times = {}  
    start_time = time.time()  

    if img.mode != 'RGB':  
        img = img.convert('RGB')  

    # Create figure with final size immediately  
    fig = plt.figure(figsize=(20, 20))  
    
    plt.imshow(img, interpolation="lanczos")  
    ax = plt.gca()  
    times['open_image'] = time.time() - start_time  

    for cell in cells:  
        bbox = cell['bbox']  
        if bbox == [0,0,0,0]:  
            continue   
        
        props = {'linewidth': 2, 'alpha': 0.3, 'linestyle': '-'}  

        if cell['column_header']:  
            props.update({'facecolor': (1, 0, 0.45), 'edgecolor': (1, 0, 0.45), 'hatch': '//////'})  
        elif cell['projected_row_header']:  
            props.update({'facecolor': (0.95, 0.6, 0.1), 'edgecolor': (0.95, 0.6, 0.1), 'hatch': '//////'})  
        else:  
            is_even = cell['row_nums'][0] % 2 == 0  
            color = (0.3, 0.74, 0.8) if is_even else (0.95, 0.9, 0.25)  
            props.update({'facecolor': color, 'edgecolor': color, 'hatch': '\\\\\\\\\\\\'})  
            
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], **props)  
        ax.add_patch(rect)  

    times['draw_cells'] = time.time() - start_time - times['open_image']  

    plt.xticks([], [])  
    plt.yticks([], [])  

    legend_elements = [  
        Patch(facecolor=(0.3, 0.74, 0.8), edgecolor=(0.3, 0.7, 0.6), label='Cell (even row)', hatch='//////', alpha=0.3),  
        Patch(facecolor=(0.95, 0.9, 0.25), edgecolor=(0.95, 0.85, 0.15), label='Cell (odd row)', hatch='//////', alpha=0.3),  
        Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45), label='Column header cell', hatch='//////', alpha=0.3)  
    ]  

    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center',   
              borderaxespad=0, fontsize=10, ncol=3)  
    
    plt.axis('off')  
    times['draw_legend'] = time.time() - start_time - times['open_image'] - times['draw_cells']  

    # Optimize saving  
    try:  
        # Use buffer_rgba() instead of deprecated tostring_rgb()  
        fig.canvas.draw()  
        buf = fig.canvas.buffer_rgba()  
        plt_image = np.asarray(buf)  
        # Convert RGBA to RGB  
        plt_image_rgb = plt_image[:, :, :3]  
        Image.fromarray(plt_image_rgb).save(out_path, optimize=True, quality=90)  
    except Exception as e:  
        print(f"Failed to save using PIL method: {str(e)}")  
        # Fallback to matplotlib saving with optimized parameters  
        plt.savefig(out_path, dpi=100, optimize=True, quality=90,   
                   bbox_inches=None, pad_inches=0.1)  
    
    plt.close(fig)  
    
    times['save_image'] = time.time() - start_time - times['open_image'] - times['draw_cells'] - times['draw_legend']  
    # Print timing summary  
    print("\nTiming Summary:")  
    print(f"Time to open the image: {times['open_image']:.3f}s")  
    print(f"Time to draw the cells: {times['draw_cells']:.3f}s")  
    print(f"Time to draw the legend: {times['draw_legend']:.3f}s")  
    print(f"Time to save the image: {times['save_image']:.3f}s")  
    print(f"Total time: {sum(times.values()):.3f}s")
def visualize_doc(doc_id, image_dir, json_dir):
    image_path = image_dir + doc_id +  "_00001.jpg"   
    json_path = json_dir + doc_id + ".json"   

    with open(json_path) as f:
        data = json.load(f)
    print (data)    
    cells = data['TSR'][0][0]['data'] # format gt
    output_name = "results/viz_gt/" + doc_id + ".png"
    img = Image.open(image_path)
    visualize_cells_table(img, cells, output_name)

    print (cells)
    
#C:\data\L3i\table-extraction\data\Yooz\DATASET\bergstrom\statements_training    
#C:\data\L3i\table-extraction\code\table\src
doc_ids = []
doc_ids.append("1_1_b7816646-d903-4165-b0e7-d39485c49000-17125-32635-1709576302517-payables-03-04-24_1_1")
doc_ids.append("1_6_b7816646-d903-4165-b0e7-d39485c49000-17125-26721-1707336995948-invoicescans-2-7-20242_6_1")

#doc_ids.append("1_b7816646-d903-4165-b0e7-d39485c49000-17125-19558-1704471085695-yooz-1-5-24_1")
#doc_ids.append("1_b7816646-d903-4165-b0e7-d39485c49000-17125-20327-1704823683377-0785-001_1")
#doc_ids.append("1_b7816646-d903-4165-b0e7-d39485c49000-17125-20708-1704919775851-17_1")
#1_1704304244596-1004503432_1_00001
#image_dir_root =  "../../../data/Yooz/DATASET/bergstrom/"
#image_dir =  image_dir_root + "statements_training/img_xml_up_to_date_filtered/"
#json_dir =  image_dir_root + "statements_training/new_GT_filtered/"

image_dir_root =  "./data/Yooz/"
image_dir =  image_dir_root + "test/img_xml/"
json_dir =  image_dir_root + "json/Bergstrom/"

for doc_id in doc_ids:
    visualize_doc(doc_id, image_dir, json_dir)

