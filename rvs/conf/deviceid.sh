#!/bin/bash
# To use this utlity call it from testcript with below example
# Example:  ./conf/deviceid.sh "conf/gpup_*.conf"
# Above command will add device_id : XXXX in all gpup_*.conf files
 
conf_to_mod=(echo $1)
#Below  configLine function I got from https://stackoverflow.com/a/54909102
function configLine {
    local OLD_LINE_PATTERN=$1; shift
    local NEW_LINE=$1; shift
    local FILE=$1
    local NEW=$(echo "${NEW_LINE}" | sed 's/\//\\\//g')
    sed -i '/'"${OLD_LINE_PATTERN}"'/{s/.*/'"${NEW}"'/;h};${x;/./{x;q100};x}' "${FILE}"
    if [[ $? -ne 100 ]] && [[ ${NEW_LINE} != '' ]]
    then
        echo "${NEW_LINE}" >> "${FILE}"
    fi
}

gpu_dev_id_list=()
device_id=0
for dir in $(ls -d /sys/class/kfd/kfd/topology/nodes/*)
do
    dev_prop_file="$dir/properties"
    gpu_dev_id=(`cat $dev_prop_file | grep device_id | awk '{ print $2 '}`)
    if [ '0' != $gpu_dev_id ]; then
        gpu_dev_id_list[$i]=$gpu_dev_id
    fi
    for j in "${gpu_dev_id_list[@]}"
    do
        #echo "j $j"
        if [ '0' != '$j' ]; then    
            d_i=$j
        fi
    #echo "d_i $d_i"
    done
done

for dev_id_line in $1; do
    configLine  "device_id" "  device_id : $d_i" $dev_id_line
done
