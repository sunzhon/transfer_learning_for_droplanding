#! /bin/pyenv python
#coding: --utf-8

from visualization import *

if __name__=='__main__':

    # Plot metrics

    subjects_ids_names = ['P_08_zhangboyuan', 'P_10_dongxuan', 'P_11_liuchunyu', 'P_13_xulibang', 'P_14_hunan', 'P_15_liuzhaoyu', 'P_16_zhangjinduo', 'P_17_congyuanqi', 'P_18_hezhonghai', 'P_19_xiongyihui', 'P_20_xuanweicheng', 'P_21_wujianing', 'P_22_zhangning', 'P_23_wangjinhong', 'P_24_liziqing']
    #subjects_ids_names.remove('P_19_xiongyihui')
    filters={'drop_value':0.4,'sort_variable':'r2','test_subject':subjects_ids_names}
    fliters={'drop_value':0.0,'sort_variable':'r2'}
    #filters={'sort_variable':'r2'}
    combination_investigation_results = [os.path.join(RESULTS_PATH, "training_testing/investigation/baseline_v14/testing_result_folders.txt")]
    metrics = get_list_investigation_metrics(combination_investigation_results)
    combination_investigation_metrics = [os.path.join(os.path.dirname(folder),"metrics.csv") for folder in combination_investigation_results]
    print(p6plot_model_accuracy(combination_investigation_metrics,fliters, ttest=False,save_fig=True))

    print('Plot sucessulflY')
