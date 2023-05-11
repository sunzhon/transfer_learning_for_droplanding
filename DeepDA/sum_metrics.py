#! /bin/pyenv python
#coding: --utf-8
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
added_path = os.path.join(current_dir,"./../assessments")
sys.path.append(added_path)
import scores as es_sc

added_path = os.path.join(current_dir,"./../vicon_imu_data_process")
sys.path.append(added_path)
import process_landing_data as pro_rd
import const


if __name__=='__main__':
    sys.path.append('./../')
    sys.path.append('./')

if(len(sys.argv)>=2):
    print("args: ", sys.argv)
    es_sc.sum_metrics(sys.argv[1:])
else:
    print("Parse is wrong")
