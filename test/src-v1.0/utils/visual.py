import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from PIL import Image
import json
import os
import numpy as np


def visualize_cells_table_old(img, cells, out_path):
    time1 = time.time()

    if img.mode != 'RGB':
        img = img.convert('RGB')

    plt.imshow(img, interpolation="lanczos")
    plt.gcf().set_size_inches(20, 20)
    ax = plt.gca()
    time2 = time.time()
    print(f"Time to open the image: {time2-time1}")
    

    for cell in cells:

        bbox = cell['bbox']
        if bbox==[0,0,0,0]:
            continue 
        props = {'linewidth': 2, 'alpha': 0.3, 'linestyle': '-'}

        if cell['column_header']:
            props['facecolor'] = (1, 0, 0.45)
            props['edgecolor'] = (1, 0, 0.45)
            props['hatch'] = '//////'

        elif cell['projected_row_header']:
            props['facecolor'] = (0.95, 0.6, 0.1)
            props['edgecolor'] = (0.95, 0.6, 0.1)
            props['hatch'] = '//////'

        else:
            props['facecolor'] = (0.3, 0.74, 0.8) if cell['row_nums'][0] % 2 == 0 else (0.95, 0.9, 0.25)
            props['edgecolor'] = (0.3, 0.7, 0.6) if cell['row_nums'][0] % 2 == 0 else (0.95, 0.9, 0.25)
            props['hatch'] = '\\\\\\\\\\\\'
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], **props)
        ax.add_patch(rect)

    time3 = time.time()
    print(f"Time to draw the cells: {time3-time2}")

    plt.xticks([], [])
    plt.yticks([], [])

    
    legend_elements = [
        Patch(facecolor=(0.3, 0.74, 0.8), edgecolor=(0.3, 0.7, 0.6), label='Cell (even row)', hatch='//////', alpha=0.3),
        Patch(facecolor=(0.95, 0.9, 0.25), edgecolor=(0.95, 0.85, 0.15), label='Cell (odd row)', hatch='//////', alpha=0.3),
        Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45), label='Column header cell', hatch='//////', alpha=0.3)
    ]

    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
                    fontsize=10, ncol=3)  
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')
    time4 = time.time()
    print(f"Time to draw the legend: {time4-time3}")
    
    # save figure 

    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()


    time5 = time.time()
    print(f"Time to save the image: {time5-time4}")

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
    #print("\nTiming Summary:")  
    #print(f"Time to open the image: {times['open_image']:.3f}s")  
    #print(f"Time to draw the cells: {times['draw_cells']:.3f}s")  
    #print(f"Time to draw the legend: {times['draw_legend']:.3f}s")  
    #print(f"Time to save the image: {times['save_image']:.3f}s")  
    #print(f"Total time: {sum(times.values()):.3f}s")
def visualize_doc(image_path, json_path, output_name):
    #print("visualize_doc->json_path=", json_path)
    with open(json_path, encoding='latin-1') as f:
        data = json.load(f)
    cells = data['TSR'][0][0]['data'] # format gt
    img = Image.open(image_path)
    visualize_cells_table(img, cells, output_name)
    
def visualize_cropped_table(image_path, json_path, output_name):
    print("visualize_cropped_table->json_path=", json_path)
    print("visualize_cropped_table->image_path=", image_path)
    with open(json_path, encoding='latin-1') as f:
        data = json.load(f)
    print("visualize_cropped_table->data=", data)
    #cells = data['TSR'][0][0]['data'] # format gt
    td = data["TD"][0][0]
    cells = data["TSR"][0][0]["data"]
    
    print("visualize_cropped_table->td=", td)
    print("visualize_cropped_table->cells=", cells)
    
    table_cells = crop_bbox_tokens(cells, td)
    
    print("visualize_cropped_table->table_cells=", table_cells)
    img = Image.open(image_path)
    visualize_cells_table(img, table_cells, output_name)
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
