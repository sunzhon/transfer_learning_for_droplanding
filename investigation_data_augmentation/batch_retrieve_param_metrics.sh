#! /bin/zsh
# env
RESULTS_PATH=`python -c 'import main; print(main.const.RESULTS_PATH)'`

while getopts ":f:l:a" opt
do
    case $opt in
        f)
            echo "参数f (folder name)的值: $OPTARG"
            result_folder=$OPTARG
            collect_sh="./retrieve_param_metrics.sh"
            file_dir="${RESULTS_PATH}/training_testing/${result_folder}"
            ${collect_sh} ${file_dir} "test_"
            ;;
        l)
            echo "参数l (list of folder name)的值: $OPTARG"
            folder_list_file=$OPTARG
            result_folder_array=($(cat "${folder_list_file}"))

            for result_folder in ${result_folder_array[@]}; do
                echo ${result_folder}

                if [ $# -gt 1 ]; then
                    filter_target=$2
                else
                    filter_target='test_'
                fi

                collect_sh="./retrieve_param_metrics.sh"
                file_dir="${RESULTS_PATH}/training_testing/${result_folder}"
                ${collect_sh} ${file_dir} ${filter_target}
            done
            ;;
        a)
            echo "参数all的值$OPTARG"

            dir_path="${RESULTS_PATH}/training_testing"
            result_folder_array=($(cat "${dir_path}/test_cases.txt"))

            for result_folder in ${result_folder_array[@]}; do
                echo ${result_folder}

                if [ $# -gt 1 ]; then
                    filter_target=$2
                else
                    filter_target='test_'
                fi

                collect_sh="./retrieve_param_metrics.sh"
                file_dir="${RESULTS_PATH}/training_testing/${result_folder}"
                ${collect_sh} ${file_dir} ${filter_target}
            done
            ;;
        ?)
            echo "未知参数, $opt, $OPTARG"
            exit 1
            ;;
        :)
            echo "没有输入任何选项 $OPTARG"
            ;;
    esac
done

