import argparse
import os
from utils import dataset as ds
import functools

print = functools.partial(print, flush=True)
import teds as eval_tool
import json
import pprint
import pandas as pd
def main(args):
    import json
    print("****** START TEDS EVALUATION ******\n\n")
    print ("args")
    print (args)
    
    teds = eval_tool.TEDS(n_jobs=1)
    
    pred_dict = eval_tool.collect_html(args.pred_dir, is_gt=False)
    true_dict = eval_tool.collect_html(args.gt_dir, is_gt = True)
    '''
    docids = []
    scores = []
    for docid in true_dict.keys():
        pred_html = pred_dict[docid]
        true_html = true_dict[docid]["html"]
        print ("pred_html=", pred_html)
        print ("true_html=", true_html)
        
        score = teds.evaluate(pred_html, true_html)
        print ("score=", score)
        docids.append(docid)
        scores.append(score)
        break
    data = {"id":docids, "score": scores}
    df = pd.DataFrame.from_dict(data)
    
    path = os.path.join(args.output_dir, args.output_file)
    
    df.to_csv(path, index = False)
    
    #return 1
    
    '''
    scores = teds.batch_evaluate(pred_dict, true_dict)
    pp = pprint.PrettyPrinter()
    #pp.pprint(scores)
    ds.ensure_dir(args.output_dir)
    path = os.path.join(args.output_dir, args.output_file)
    #ds.put_content_to_file(scores, path)
    print("length of scores:", len(scores) , "final average score:", sum(scores.values()) / len(scores.values()))
    print("****** END TEDS EVALUATION ******\n\n")
    docids = [k.replace(".html", "") for k in scores.keys()]
    data = {"id":docids, "score": scores.values()}
    df = pd.DataFrame.from_dict(data)    
    path = os.path.join(args.output_dir, args.output_file)    
    df.to_csv(path, index = False)
    
    return 1    
if __name__ == "__main__":
    
    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_dir',
        type=str,
        default='data/ground-truth-wo-text',
        help=''
    )
    parser.add_argument(
        '--pred_dir',
        type=str,
        default='results/html/FinTabNet-TSR-wo-text-small',
        help=''
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/evaluation-small',
        help=''
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='teds_scores-wo-text-small.csv',
        help=''
    )
    
    FLAGS = parser.parse_args()
    print("****** PARAMS ******\n\n")
    print (FLAGS)  
    main(FLAGS)
    