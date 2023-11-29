#! /bin/zsh
# env
RESULTS_PATH=`python -c 'import main; print(main.const.RESULTS_PATH)'`

# argument
if [ $# -gt 0 ]; then
	result_folder=$1
	collect_sh="./retrieve_param_metrics.sh"
	file_dir="${RESULTS_PATH}/training_testing/${result_folder}"
	${collect_sh} ${file_dir} "test_"
else
	# read test_result folder from a txt file
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

fi


