import argparse
import os
from utils import dataset as ds
import functools
print = functools.partial(print, flush=True)
import pandas as pd
import matplotlib.pyplot as plt

def save_viz(args, bins, labels):
    print("****** SAVE ******\n\n")
    plt.clf()
    datas = ds.metadata_load (args.result_dir, args.result_file)
    datas['level'] = datas['score'] * 100
    datas['bins'] = pd.cut(datas['level'],bins=bins, labels=labels)
    chart_df = datas.groupby(['bins']).size()
    print (chart_df)
    print("RESULTS:", datas)
    print("****** END EVALUATION ******\n\n")
    #chart_df.plot.pie(y='mass', subplots=True,figsize=(8, 3))
    #plot = chart_df.plot.pie(subplots=True,figsize=(8, 3))
    chart_df.plot(kind='pie', y='bins', autopct='%1.0f%%',title='Chart by ' + args.score_type + ' Score')
    path = os.path.join(args.result_dir, args.result_file + args.suff + ".jpg")
    plt.savefig(path)
    path_csv = os.path.join(args.result_dir, args.result_file + args.suff + ".csv")
    
    sorted_df = datas.sort_values(by=['level'], ascending=True)
    sorted_df.to_csv(path_csv)
    return 1   
    
def main(args):
    print("****** START EVALUATION ******\n\n")
    print ("args")
    print (args)
    
    
    bins=[0, 25, 50, 75, 100]
    labels=["0-25","25-50","50-75","75-100"]
    args.suff = "-S4"
    save_viz(args, bins, labels)
    
    bins=[0, 25, 50, 70, 80, 90, 100]
    labels=["0-25","25-50","50-70","70-80","80-90","90-100"]
    args.suff = "-S6"
    save_viz(args, bins, labels)
    
    
    bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels=["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"]
    args.suff = "-S10"
    save_viz(args, bins, labels)
    
    
    return 1
    
if __name__ == "__main__":
    
    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_dir',
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
        '--score_type',
        type=str,
        default='TEDS',
        help=''
    )
    
    
    FLAGS = parser.parse_args()
    print("****** PARAMS ******\n\n")
    print (FLAGS)  
    main(FLAGS)
    