#! /bin/zsh

if [ $# -gt 0 ]; then
    testing_folders=$1
else
    testing_folders='.'
fi

if [ $# -gt 1 ]; then
    filter_target=$2
else
    filter_target='test_'
fi

list_hyper_files=($(find $testing_folders -name hyperparams.yaml))

data_file="$testing_folders/testing_result_folders.txt"
touch $data_file
#echo "" > $data_file
echo "Sensor configurations\tLSTM units\tsyn_features_labels\tuse_frame_index\tlanding_manners\testimated_variables\ttraining_testing_folders" > $data_file
echo "START TO COLLECT TEST DATA"

#all_sensor_configs=( 'T' 'S' 'F' 'W' 'C'  'FS' 'FT' 'FW' 'FC' 'ST' 'SW' 'SC' 'TW' 'TC' 'WC' 'FST' 'FSW' 'FSC' 'FTW' 'FTC' 'FWC' 'STW' 'STC' 'SWC' 'TWC' 'FSTW' 'FSTC' 'FSWC' 'FTWC' 'STWC' 'FSTWC')
#all_sensor_configs=('FSTWCF' 'FSTWCS' 'FSTWCT' 'FSTWCFS' 'FSTWCFT' 'FSTWCST' 'FSTWCFST')
all_sensor_configs=( 'T' 'S' 'F' 'W' 'C'  'FS' 'FT' 'FW' 'FC' 'ST' 'SW' 'SC' 'TW' 'TC' 'WC' 'FST' 'FSW' 'FSC' 'FTW' 'FTC' 'FWC' 'STW' 'STC' 'SWC' 'TWC' 'FSTW' 'FSTC' 'FSWC' 'FTWC' 'STWC' 'FSTWC' 'FSTWCF' 'FSTWCS' 'FSTWCT' 'FSTWCFS' 'FSTWCFT' 'FSTWCST' 'FSTWCFST')

for hyper_file in ${list_hyper_files}; do 
    str=${hyper_file%/*}
    str1=${str##*/}
    echo $str1
    echo ${filter_target}

    if [[ $str1 =~ ${filter_target}[0-9]+ ]];then # list training_* or testing_*
        echo ${hyper_file}
        folder_path=$(cd $(dirname $hyper_file); pwd)
        echo ${folder_path}
        lstm=$(awk -F"[ :-]+" '$1~/lstm_units/{print $2}' $hyper_file | grep -o -E "\w+")
        sensors_fields=$(awk -F"[ :-]+" '$2~/Accel_X/{array[$2]++}END{for(i in array)print i}' $hyper_file)
        syn_features_labels=$(awk -F"[ :-]+" '$1~/syn_features_labels/{print $2}' $hyper_file)
        use_frame_index=$(awk -F"[ :-]+" '$1~/use_frame_index/{print $2}' $hyper_file)
        landing_manners=$(awk -F"[ :-]+" '$1~/landing_manner/{print $2}' $hyper_file)
        #estimated_variables=$(awk -F"[ :-]+" '$2~/KNEE_MOMENT|GRF/{array[$2]++}END{for( i in array) if i==print "["i"]"}' $hyper_file)
        estimated_variables=$(awk -F"[ :-]+" '$2~/KNEE_MOMENT_X|GRF_Z/{array[$2]++}END{for(i in array){if(i~/KNEE_MOMENT_X/){print "[KFM]"}else if(i~/KNEE_MOMENT_Y/){print "[KAM]"}else if(i~/GRF_Z/){print "[GRF]"}}}' $hyper_file)



        echo "lstm units: ${lstm}" 
        #echo "sensor fields:\n${sensors_fields}"
        sensor_configs=$(echo $sensors_fields | grep -o -E "[A-Z]{2,10}")
        echo "sensor configs:\n$sensor_configs"
        combined_sensor_config_name=$(echo $sensor_configs | grep -o -E "^[A-Z]|(\s[A-Z])+" | grep -o -E "[A-Z]+" | tr -d '\n')
        echo "combined sensor configuration name: $combined_sensor_config_name"
        for a_sensor_config in ${all_sensor_configs[@]}; do
            n_a_sensor_config=$(echo $a_sensor_config | wc -m)
            n_a_sensor_config=$[$n_a_sensor_config-1]
            temp=$(echo $combined_sensor_config_name | grep -E -o "([$a_sensor_config]){$n_a_sensor_config}") 
            if [ "$temp" = "$combined_sensor_config_name" ]; then
                real_sensor_config_name=$a_sensor_config
                echo "number of sensor name characters" $n_a_sensor_config
                echo "temp combined sensor name: $temp"
                echo "combined sensor config name: $combined_sensor_config_name"
                echo "real sensor config:" $real_sensor_config_name
                #echo "${real_sensor_config_name}\t${lstm}\t${folder_path}" >> $data_file
                echo "${real_sensor_config_name}\t${lstm}\t${syn_features_labels}\t${use_frame_index}\t${landing_manners}\t${estimated_variables}\t${folder_path}" >> $data_file
            fi
        done
    fi
done

sed -i -e "s/'true'/true/g" $data_file
sed -i -e "s/'false'/false/g" $data_file
sed -i -e "s/\[GRF\]/GRF/g" $data_file
sed -i -e "s/\[KFM\]/KFM/g" $data_file
sed -i -e "s/\[KAM\]/KAM/g" $data_file


echo "END TO COLLECT TEST DATA"

