"""
Simple Framework for Incremental Learning

Authors: PHAM Tri Cong*+
Affiliation: *L3i, La Rochelle University, La Rochelle, 17000, France.
             +Yooz, 1 Rue Fleming, La Rochelle, 17000, France.
Corresponding Email: cong.pham@univ-lr.fr
Date: Oct 20, 2022

"""

import argparse
import os
import numpy as np
import pandas as pd
import re
import pickle
import csv
from PIL import Image
'''
common function
'''
def select_type_to_max(y_true, select_type):
    y_count = np.unique(y_true, return_counts=True)
    min_number = min(y_count[1])
    if min_number < 15:
        min_number = 15
    if select_type > 1:
        return min_number * select_type
    return min_number
def ensure_dir_for_file(file_path):
    directory = os.path.dirname(file_path)
    ensure_dir(directory)
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def put_content_to_file(data, path):
    with open(path, 'wb') as ofile:
        pickle.dump(data, ofile)
def get_content_from_file(path):
    with open(path, 'rb') as data:
        return pickle.load(data)
def metadata_load (data_dir, dtype, encoding='latin-1'):
    sep=","
    path = os.path.join(data_dir, dtype + ".csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep=sep, encoding = encoding)
def metadata_save(data_dir, df, dtype = "", encoding='latin-1'):
    path = os.path.join(data_dir, dtype + ".csv")
    df.to_csv(path, encoding = encoding, index=False)
def filter_datas(datas):
    ret = datas[datas["class"] > -1]
    print("filter_datas->from len[", len(datas), "]", "to len[", len(ret), "]")
    return ret
def filter_datas_new(datas):
    ret = datas[datas["class"] == -1]
    print("filter_datas->from len[", len(datas), "]", "to len[", len(ret), "]")
    return ret
def text_csv_load (data_dir, dtype, sep = ",", names=None, encoding='latin-1'):
    #sep=","
    file_path = os.path.join(data_dir, dtype)
    if names is None:
        return pd.read_csv(file_path, sep = sep, encoding = encoding)
    return pd.read_csv(file_path, sep = sep, encoding = encoding, names = names)
def text_csv_save(data_dir, datas, type = "", encoding='latin-1'):
    datas["text"] = datas["path"].map(lambda x: "" if (not x) else text_data_load(x))
    text_csv_save_with_text(data_dir, datas, type, encoding)
def text_csv_save_with_text(data_dir, datas, type = "", encoding='latin-1'):
    #print("text_csv_save")
    #print(datas)
    path = os.path.join(data_dir, type + ".csv")
    csvfile = open(path, "w",newline='', encoding = encoding)
    csvwriter = csv.writer(csvfile)
    cols = ["id","class", "text"]
    csvwriter.writerow(cols)
    for ind, data in datas.iterrows():
        text = data["text"]
        if not text:
            print ("error text is null")
            print (data)
        else:
            csvwriter.writerow([data["id"], data["class"], text])
    csvfile.close()
def text_data_load (path, encoding='latin-1'):
    text = ""
    with open(path,'r', encoding = encoding) as file:
        vec = file.readlines()            
        vec_proc = [remove_mystopwords(t) for  t in vec]
        text = " ".join(vec_proc).replace('\n',' ').strip()
    return text
def remove_mystopwords(sentence):
    tokens = sentence.split(" ")
    tokens_filtered= [re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ: ]', ' ', word)  for word in tokens]
    return (" ").join(tokens_filtered)

'''
for French text replay
'''
def yooz_fr_text_meta_data_load (label_data_dir, text_data_dir, dtype, map_labels, keep_label = False):
    datas = yooz_csv_meta_load (label_data_dir, dtype, sep=";")
    fmap = {}
    folder_scan_by_ext(fmap, text_data_dir, ".txt")
    if keep_label:
        datas["label"] = datas["class"]
    datas["class"] = datas["class"].map(lambda x: map_labels[x])
    datas["path"] = datas["id"].map(lambda x: fmap[x] if (x in fmap) else None)
    ret = datas[datas["path"].notnull()]
    ret_null = datas[datas["path"].isnull()]
    return ret, ret_null
def yooz_csv_meta_load(data_dir, dtype, sep="	"):
    file_path = os.path.join(data_dir, dtype + ".txt")
    return pd.read_csv(file_path, sep=sep, encoding='latin-1', names=["id","class"], converters={'id':yooz_convert_to_id})

def folder_scan_by_ext(datas, data_dir, ext = ".tif"):
    obj = os.scandir(data_dir)
    for entry in obj :
        if entry.is_file():
            fname = entry.name.lower()
            if not fname.endswith(ext):
                continue
            fileid = yooz_convert_to_id(entry.name)
            path = os.path.join(data_dir, entry.name)
            datas[fileid] = path
    return datas

def yooz_convert_to_id(x):
    return os.path.splitext(x.lower())[0]
'''
for French Image replay
'''
def shoebox_data_load(data_dir, map_labels, keep_label = False):
    print("****** READ LABELS DATA ******")
    label_train_datas = shoebox_load_labels(data_dir,"ShoeboxLearn")
    label_test_datas = shoebox_load_labels(data_dir,"ShoeboxTest")
    #check images of training
    print("****** CHECKING IMAGE ******")    
    data_dir_img = os.path.join(data_dir,"learn")
    train_datas, train_datas_no_img = shoebox_data_pre_encode(label_train_datas, data_dir_img, map_labels, keep_label)
    data_dir_img = os.path.join(data_dir,"test")
    test_datas, test_datas_no_img = shoebox_data_pre_encode(label_test_datas, data_dir_img, map_labels, keep_label)
    print("****** END LOAD DATA ******")
    return train_datas, test_datas, train_datas_no_img, test_datas_no_img
  
def shoebox_data_pre_encode(datas, data_dir, map_labels, keep_label = True):
    mapping = {}
    obj = os.scandir(data_dir)
    print("Files and Directories in '% s':" % data_dir)
    folders = [data_dir]
    for entry in obj :
        if entry.is_dir():
            folders.append(os.path.join(data_dir, entry.name))
    for folder in folders :
        shoebox_data_pre_encode_folder(mapping, folder)
    #print (mapping)
    #print (datas)
    #map_labels = map_labels_to_ids()
    if keep_label:
        datas["label"] = datas["class"]
    datas["class"] = datas["class"].map(lambda x: map_labels[x])
    datas["path"] = datas["id"].map(lambda x: mapping[x] if (x in mapping) else None)
    ret = datas[datas["path"].notnull()]
    ret_null = datas[datas["path"].isnull()]
    return ret, ret_null
def shoebox_data_pre_encode_folder(datas, data_dir, ext = ".tif"):
    obj = os.scandir(data_dir)
    for entry in obj :
        if entry.is_file():
            fname = entry.name.lower()
            if not fname.endswith(ext):
                continue
            fileid = yooz_convert_to_id(entry.name)
            path = os.path.join(data_dir, entry.name)
            datas[fileid] = path
    return datas
def shoebox_load_labels(data_dir, dtype, sep="	"):
    file_path = os.path.join(data_dir, dtype + ".txt")
    return pd.read_csv(file_path, sep=sep, encoding='latin-1', names=["id","class"], converters={'id':yooz_convert_to_id})



def yooz_fr_map_labels_to_ids(dataset):
    if dataset == "full":
        return yooz_fr_map_labels_to_ids_full()
    elif dataset == "medium":
        return yooz_fr_map_labels_to_ids_medium()
    elif dataset == "small":
        return yooz_fr_map_labels_to_ids_small()
    elif dataset == "tiny":
        return yooz_fr_map_labels_to_ids_tiny()
    return None    
def yooz_fr_map_labels_to_ids_full():
    return {  
        "ACNT_INVOICE"          : 0,
        "ACNT_DUES"             : 1,
        "ACNT_EXP"              : 2,
        "ACNT_PAYSLIP"          : 3,
        "ACNT_INVOICELIST"      : 4,
        "ACNT_CALLCAPITAL"      : 5,
        "ACNT_TAXNOTICE"        : 6,
        "ACNT_NOTICEDEBIT"      : 7,
        "ACNT_NOTICEPAY"        : 8,
        "ACNT_STATMNT"          : 9,
        "ACNT_DUCS"             : 10,
        "ACNT_NOTIFICATION"     : 11,
        "ACNT_FINALDEMAND"      : 12,
        "ACNT_FINE"             : 13,
        "ACNT_DEBITMANDATE"     : 14, #11 samples
        "BANK_STATMNT"          : 15,
        "BANK_CREDITSTATMNT"    : 16,
        "BANK_BES"              : 17,
        "BANK_ACCOUNTID"        : 18,
        "BANK_PAYABLENOTE"      : 19,
        "BANK_DEBITSTATMNT"     : 20, #6 samples
        "BUSN_GDRCPT"           : 21,
        "BUSN_STATMNT"          : 22,
        "BUSN_DELIVNOTE"        : 23,
        "BUSN_ORDER"            : 24,
        "BUSN_QUOTE"            : 25,
        "BUSN_CONTRACT"         : 26,
        "BUSN_WORK"             : 27,
        "MAIL_BANK_PAYABLENOTE" : 28,
        "MAIL"                  : 29,
        "MAIL_ACNT_DUES"        : 30, #3 samples
        "LGAL_CONTRACT"         : 31,
        "LGAL_REPORT_GA"        : 32,
        "LGAL_WORKSTOP"         : 33,
        "LGAL_KBIS"             : 34,
        "INSUR_CONTRACT"        : 35,
        "INSUR_DAILYALLOWANCE"  : 36, #8 samples
        "CGV"                   : 37,
        "CHQ"                   : 38,
        "VERSO"                 : 39,
        "ID"                    : 40, #only 5 samples, test sample is not good
        "BANK_OPPOSITION"       : 41, #only 2 samples
        "MAIL_RETURNCOUPON"     : 42, #only 3 samples
        "MAIL_ACNT_INVOICE"     : 43, #only 2 samples
        "MAIL_BUSN_CONTRACT"    : 44, #only 2 samples
        "MAIL_ACNT_CALLCAPITAL" : 45, #only 1 samples
        "LGAL_CV"               : 46, #only 1 samples        
    }
def yooz_fr_map_labels_to_ids_medium():
    return {  
        "ACNT_INVOICE"          : 0,
        "ACNT_DUES"             : 1,
        "ACNT_EXP"              : 2,
        "ACNT_PAYSLIP"          : 3,
        "ACNT_INVOICELIST"      : 4,
        "ACNT_CALLCAPITAL"      : 5,
        "ACNT_TAXNOTICE"        : 6,
        "ACNT_NOTICEDEBIT"      : 7,
        "ACNT_NOTICEPAY"        : 8,
        "ACNT_STATMNT"          : 9,
        "ACNT_DUCS"             : 10,
        "ACNT_NOTIFICATION"     : 11,
        "ACNT_FINALDEMAND"      : 12,
        "ACNT_FINE"             : 13,
        "ACNT_DEBITMANDATE"     : 14, #11 samples
        "BANK_STATMNT"          : 15,
        "BANK_CREDITSTATMNT"    : 16,
        "BANK_BES"              : 17,
        "BANK_ACCOUNTID"        : 18,
        "BANK_PAYABLENOTE"      : 19,
        "BANK_DEBITSTATMNT"     : 20, #6 samples
        "BUSN_GDRCPT"           : 21,
        "BUSN_STATMNT"          : 22,
        "BUSN_DELIVNOTE"        : 23,
        "BUSN_ORDER"            : 24,
        "BUSN_QUOTE"            : 25,
        "BUSN_CONTRACT"         : 26,
        "BUSN_WORK"             : 27,
        "MAIL_BANK_PAYABLENOTE" : 28,
        "MAIL"                  : 29,
        "MAIL_ACNT_DUES"        : 30, #3 samples
        "LGAL_CONTRACT"         : 31,
        "LGAL_REPORT_GA"        : 32,
        "LGAL_WORKSTOP"         : 33,
        "LGAL_KBIS"             : 34,
        "INSUR_CONTRACT"        : 35,
        "INSUR_DAILYALLOWANCE"  : 36, #8 samples
        "CGV"                   : 37,
        "CHQ"                   : 38,
        "VERSO"                 : 39,
        "ID"                    : 40, #only 5 samples, test sample is not good
        "BANK_OPPOSITION"       : -1, #only 2 samples
        "MAIL_RETURNCOUPON"     : -1, #only 3 samples
        "MAIL_ACNT_INVOICE"     : -1, #only 2 samples
        "MAIL_BUSN_CONTRACT"    : -1, #only 2 samples
        "MAIL_ACNT_CALLCAPITAL" : -1, #only 1 samples
        "LGAL_CV"               : -1, #only 1 samples        
    }
def yooz_fr_map_labels_to_ids_small():
    return {  
        "ACNT_INVOICE"          : 0,
        "ACNT_DUES"             : 1,
        "ACNT_EXP"              : 2,
        "ACNT_PAYSLIP"          : 3,
        "ACNT_INVOICELIST"      : 4,
        "ACNT_CALLCAPITAL"      : 5,
        "ACNT_TAXNOTICE"        : 6,
        "ACNT_NOTICEDEBIT"      : 7,
        "ACNT_NOTICEPAY"        : 8,
        "ACNT_STATMNT"          : 9,
        "ACNT_DUCS"             : 10,
        "ACNT_NOTIFICATION"     : 11,
        "ACNT_FINALDEMAND"      : 12,
        "ACNT_FINE"             : 13,
        "BANK_STATMNT"          : 14,
        "BANK_CREDITSTATMNT"    : 15,
        "BANK_BES"              : 16,
        "BANK_ACCOUNTID"        : 17,
        "BANK_PAYABLENOTE"      : 18,
        "BUSN_GDRCPT"           : 19,
        "BUSN_STATMNT"          : 20,
        "BUSN_DELIVNOTE"        : 21,
        "BUSN_ORDER"            : 22,
        "BUSN_QUOTE"            : 23,
        "BUSN_CONTRACT"         : 24,
        "BUSN_WORK"             : 25,
        "MAIL_BANK_PAYABLENOTE" : 26,
        "MAIL"                  : 27,
        "LGAL_CONTRACT"         : 28,
        "LGAL_REPORT_GA"        : 29,
        "LGAL_WORKSTOP"         : 30,
        "LGAL_KBIS"             : 31,
        "INSUR_CONTRACT"        : 32,
        "CGV"                   : 33,
        "CHQ"                   : 34,
        "VERSO"                 : 35,
        "ID"                    : -1, #only 5 samples, test sample is not good
        "ACNT_DEBITMANDATE"     : -1, #11 samples
        "BANK_DEBITSTATMNT"     : -1, #6 samples
        "MAIL_ACNT_DUES"        : -1, #3 samples
        "INSUR_DAILYALLOWANCE"  : -1, #8 samples
        "BANK_OPPOSITION"       : -1, #only 2 samples
        "MAIL_RETURNCOUPON"     : -1, #only 3 samples
        "MAIL_ACNT_INVOICE"     : -1, #only 2 samples
        "MAIL_BUSN_CONTRACT"    : -1, #only 2 samples
        "MAIL_ACNT_CALLCAPITAL" : -1, #only 1 samples
        "LGAL_CV"               : -1, #only 1 samples        
    }    
def yooz_fr_map_labels_to_ids_tiny():
    return {
        "ACNT_INVOICE" : 0,
        "BANK_STATMNT" : 1,
        "BUSN_GDRCPT" : 2,
        "BUSN_STATMNT" : 3,
        "BANK_CREDITSTATMNT" : 4,
        "BANK_BES" : 5,
        "ACNT_DUES" : 6,
        "ACNT_EXP" : 7,
        "ACNT_PAYSLIP" : 8,
        "ACNT_INVOICELIST" : 9,
        "BUSN_DELIVNOTE" : 10,
        "BUSN_ORDER" : 11,
        "BUSN_QUOTE" : 12,
        "ACNT_CALLCAPITAL" : 13,
        "BANK_ACCOUNTID" : 14,
        "ACNT_NOTICEDEBIT" : 15,
        "ACNT_TAXNOTICE" : 16,
        "BANK_PAYABLENOTE" : 17,
        "LGAL_CONTRACT" : 18,
        "ACNT_NOTICEPAY" : 19,
        "CGV" : 20,
        "ACNT_STATMNT" : 21,
        "BUSN_CONTRACT" : 22,
        "ACNT_DUCS" : -1,
        "ACNT_NOTIFICATION" : -1,
        "ACNT_FINALDEMAND" : -1,
        "INSUR_CONTRACT" : -1,
        "ACNT_FINE" : -1,
        "MAIL_BANK_PAYABLENOTE" : -1,
        "BUSN_WORK" : -1,
        "CHQ" : -1,
        "MAIL" : -1,
        "LGAL_REPORT_GA" : -1,
        "LGAL_WORKSTOP" : -1,
        "VERSO" : -1,
        "LGAL_KBIS" : -1,
        "ID" : -1, #only 5 samples, test sample is not good
        "ACNT_DEBITMANDATE" : -1, #11 samples
        "BANK_DEBITSTATMNT" : -1, #6 samples
        "MAIL_ACNT_DUES" : -1, #3 samples
        "INSUR_DAILYALLOWANCE" : -1, #8 samples
        "BANK_OPPOSITION" : -1, #only 2 samples
        "MAIL_RETURNCOUPON" : -1, #only 3 samples
        "MAIL_ACNT_INVOICE" : -1, #only 2 samples
        "MAIL_BUSN_CONTRACT" : -1, #only 2 samples
        "MAIL_ACNT_CALLCAPITAL" : -1, #only 1 samples
        "LGAL_CV" : -1, #only 1 samples        
    }    
           
def yooz_en_map_labels_to_ids(dataset):
    if dataset == "full":
        return yooz_en_map_labels_to_ids_full()
    elif dataset == "medium":
        return yooz_en_map_labels_to_ids_medium()
    elif dataset == "small":
        return yooz_en_map_labels_to_ids_small()
    elif dataset == "tiny":
        return yooz_en_map_labels_to_ids_tiny()
    elif dataset == "cfull":
        return yooz_en_map_labels_to_ids_cdip(0)
    elif dataset == "csmall":
        return yooz_en_map_labels_to_ids_cdip(0)
    elif dataset == "ctiny25":
        return yooz_en_map_labels_to_ids_cdip(4)
    elif dataset == "ctiny50":
        return yooz_en_map_labels_to_ids_cdip(8)
    elif dataset == "ctiny75":
        return yooz_en_map_labels_to_ids_cdip(12)
    return None    
 
def yooz_en_map_labels_to_ids_full():
    return { 
            'ACNT_INVOICE'      : 0, #945
            'BUSN_ORDER'        : 1, #429
            'BUSN_DELIVNOTE'    : 2, #387
            'ACNT_INVOICELIST'  : 3, #298
            'LGAL_CERTITIT'     : 4, #286
            'LGAL_CERTITIT_V'   : 5, #275
            'MAIL_ACNT_INVOICE' : 6, #193
            'ACNT_DUES'         : 7, #95
            'BUSN_STATMNT'      : 8, #94
            'OTHER'             : 9, #71
            'MAIL'              : 10, #65
            'ACNT_TAXNOTICE'    : 11, #60
            'BUSN_WORK'         : 12, #57
            'ACNT_STATMNT'      : 13, #49
            'BUSN_QUOTE'        : 14, #39
            'EMPTY'             : 15, #36
            'LGAL_CERTITIT_DEMAND': 16, #32
            'BANK_STATMNT'      : 17, #31
            'UNK'               : 18, #29
            'ACNT_EXP'          : 19, #21
            'BUSN_CONTRACT'     : 20, #19
            'BANK_CREDITSTATMNT': 21, #18
            'INSUR_CONTRACT'    : 22, #18
            'CGV'               : 23, #16
            'ACNT_NOTICEDEBIT'  : 24, #16
            'ACNT_CALLCAPITAL'  : 25, #13
            'ACNT_NOTICEPAY'    : 26, #7
            'ACNT_CREDIT'       : 27, #7
            'LGAL_ATTENDEES_GA' : 28, #5
            'BANK_ACCOUNTID'    : 29, #4
            'LGAL_CONTRACT'     : 30, #4
            'ACNT_FINALDEMAND'  : 31, #2
            'CHQ'               : 32, #2
            'ACNT_NOTIFICATION' : 33, #2
            'ACNT_FINE'         : 34, #1
            'MAIL_BUSN_ORDER'   : 35, #1
            'ACNT_DEBITMANDATE' : 36, #0
            'MAIL_BUSN_CONTRACT': 37, #0
            'MAIL_ACNT_NOTICEPAY': 38, #0
            'ACNT_SAVING'       : 39, #0
        }
def yooz_en_map_labels_to_ids_medium():
    return { 
            'ACNT_INVOICE'      : 0, #945
            'BUSN_ORDER'        : 1, #429
            'BUSN_DELIVNOTE'    : 2, #387
            'ACNT_INVOICELIST'  : 3, #298
            'LGAL_CERTITIT'     : 4, #286
            'LGAL_CERTITIT_V'   : 5, #275
            'MAIL_ACNT_INVOICE' : 6, #193
            'ACNT_DUES'         : 7, #95
            'BUSN_STATMNT'      : 8, #94
            'OTHER'             : 9, #71
            'MAIL'              : 10, #65
            'ACNT_TAXNOTICE'    : 11, #60
            'BUSN_WORK'         : 12, #57
            'ACNT_STATMNT'      : 13, #49
            'BUSN_QUOTE'        : 14, #39
            'EMPTY'             : 15, #36
            'LGAL_CERTITIT_DEMAND': 16, #32
            'BANK_STATMNT'      : 17, #31
            'UNK'               : 18, #29
            'ACNT_EXP'          : 19, #21
            'BUSN_CONTRACT'     : 20, #19
            'BANK_CREDITSTATMNT': 21, #18
            'INSUR_CONTRACT'    : 22, #18
            'CGV'               : 23, #16
            'ACNT_NOTICEDEBIT'  : 24, #16
            'ACNT_CALLCAPITAL'  : 25, #13
            'ACNT_NOTICEPAY'    : 26, #7
            'ACNT_CREDIT'       : 27, #7
            'LGAL_ATTENDEES_GA' : 28, #5
            'BANK_ACCOUNTID'    : -1, #4
            'LGAL_CONTRACT'     : -1, #4
            'ACNT_FINALDEMAND'  : -1, #2
            'CHQ'               : -1, #2
            'ACNT_NOTIFICATION' : -1, #2
            'ACNT_FINE'         : -1, #1
            'MAIL_BUSN_ORDER'   : -1, #1
            'ACNT_DEBITMANDATE' : -1, #0
            'MAIL_BUSN_CONTRACT': -1, #0
            'MAIL_ACNT_NOTICEPAY': -1, #0
            'ACNT_SAVING'       : -1, #0
        }
def yooz_en_map_labels_to_ids_small():
    return { 
            'ACNT_INVOICE'      : 0, #945
            'BUSN_ORDER'        : 1, #429
            'BUSN_DELIVNOTE'    : 2, #387
            'ACNT_INVOICELIST'  : 3, #298
            'LGAL_CERTITIT'     : 4, #286
            'LGAL_CERTITIT_V'   : 5, #275
            'MAIL_ACNT_INVOICE' : 6, #193
            'ACNT_DUES'         : 7, #95
            'BUSN_STATMNT'      : 8, #94
            'OTHER'             : 9, #71
            'MAIL'              : 10, #65
            'ACNT_TAXNOTICE'    : 11, #60
            'BUSN_WORK'         : 12, #57
            'ACNT_STATMNT'      : 13, #49
            'BUSN_QUOTE'        : 14, #39
            'EMPTY'             : 15, #36
            'LGAL_CERTITIT_DEMAND': 16, #32
            'BANK_STATMNT'      : 17, #31
            'UNK'               : 18, #29
            'ACNT_EXP'          : 19, #21
            'BUSN_CONTRACT'     : 20, #19
            'BANK_CREDITSTATMNT': 21, #18
            'INSUR_CONTRACT'    : 22, #18
            'CGV'               : 23, #16
            'ACNT_NOTICEDEBIT'  : 24, #16
            'ACNT_CALLCAPITAL'  : 25, #13
            'ACNT_NOTICEPAY'    : -1, #7
            'ACNT_CREDIT'       : -1, #7
            'LGAL_ATTENDEES_GA' : -1, #5
            'BANK_ACCOUNTID'    : -1, #4
            'LGAL_CONTRACT'     : -1, #4
            'ACNT_FINALDEMAND'  : -1, #2
            'CHQ'               : -1, #2
            'ACNT_NOTIFICATION' : -1, #2
            'ACNT_FINE'         : -1, #1
            'MAIL_BUSN_ORDER'   : -1, #1
            'ACNT_DEBITMANDATE' : -1, #1
            'ACNT_DEBITMANDATE' : -1, #0
            'MAIL_BUSN_CONTRACT': -1, #0
            'MAIL_ACNT_NOTICEPAY': -1, #0
            'ACNT_SAVING'       : -1, #0
        }
def yooz_en_map_labels_to_ids_tiny():
    return { 
            'ACNT_INVOICE'      : 0, #945
            'BUSN_ORDER'        : 1, #429
            'BUSN_DELIVNOTE'    : 2, #387
            'ACNT_INVOICELIST'  : 3, #298
            'LGAL_CERTITIT'     : 4, #286
            'LGAL_CERTITIT_V'   : 5, #275
            'MAIL_ACNT_INVOICE' : 6, #193
            'ACNT_DUES'         : 7, #95
            'BUSN_STATMNT'      : 8, #94
            'OTHER'             : 9, #71
            'MAIL'              : 10, #65
            'ACNT_TAXNOTICE'    : 11, #60
            'BUSN_WORK'         : 12, #57
            'ACNT_STATMNT'      : 13, #49
            'BUSN_QUOTE'        : 14, #39
            'EMPTY'             : 15, #36
            'LGAL_CERTITIT_DEMAND': 16, #32
            'BANK_STATMNT'      : 17, #31
            'UNK'               : 18, #29
            'ACNT_EXP'          : 19, #21
            'BUSN_CONTRACT'     : -1, #19
            'BANK_CREDITSTATMNT': -1, #18
            'INSUR_CONTRACT'    : -1, #18
            'CGV'               : -1, #16
            'ACNT_NOTICEDEBIT'  : -1, #16
            'ACNT_CALLCAPITAL'  : -1, #13
            'ACNT_NOTICEPAY'    : -1, #7
            'ACNT_CREDIT'       : -1, #7
            'LGAL_ATTENDEES_GA' : -1, #5
            'BANK_ACCOUNTID'    : -1, #4
            'LGAL_CONTRACT'     : -1, #4
            'ACNT_FINALDEMAND'  : -1, #2
            'CHQ'               : -1, #2
            'ACNT_NOTIFICATION' : -1, #2
            'ACNT_FINE'         : -1, #1
            'MAIL_BUSN_ORDER'   : -1, #1
            'ACNT_DEBITMANDATE' : -1, #1
            'ACNT_DEBITMANDATE' : -1, #0
            'MAIL_BUSN_CONTRACT': -1, #0
            'MAIL_ACNT_NOTICEPAY': -1, #0
            'ACNT_SAVING'       : -1, #0
        }
def rvl_cdip_labels():
    return ["letter",
        "form",
        "email",
        "handwritten",
        "advertisement",
        "scientific report",
        "scientific publication",
        "specification",
        "file folder",
        "news article",
        "budget",
        "invoice",
        "presentation",
        "questionnaire",
        "resume",
        "memo"]
def yooz_en_map_labels_to_ids_cdip(num):
    names = rvl_cdip_labels()
    ret = {}
    for ind, name in enumerate(names):
        cid = ind
        if (num > 0) and (num <= ind):
            cid = -1
        ret[name] = cid
        
    return ret    