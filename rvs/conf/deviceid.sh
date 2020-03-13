#!/bin/bash
# To use this utlity call it from testcript with below example from build/bin
# Example:  ./conf/deviceid.sh "conf/gpup_*.conf"
# Above command will add device_id : XXXX in all gpup_*.conf files
# To change any single file use as below example
# ./conf/deviceid.sh conf/gpup_1.conf
set -e
 
#Below  configLine function I got from https://stackoverflow.com/a/54909102
# configLine will check for device_id string in current file and replace or
# add device_id if not present
function configLine {
    local OLD_LINE_PATTERN=$1; shift
    local NEW_LINE=$1; shift
    local FILE=$1
    local NEW=$(echo "${NEW_LINE}" | sed 's/\//\\\//g')
    sed -i '/'"${OLD_LINE_PATTERN}"'/{s/.*/'"${NEW}"'/;h};${x;/./{x;q100};x}' "${FILE}"
}

gpu_dev_id_list=()
device_id=0
# In below loop device_id is got extracted by reading kfd gpu nodes
# properties file and add it to gpu_dev_id_list array. Here it will
# read CPU integtared GPU and discrete GPU. So after that check
# for device id in array for non zero for discrete GPU and assign it to
# d_i which is final extracted value for device_id that is going to add
# to conf file 
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

# Below code is to get available device and update device: with correct value 
# in .conf files passed as argument to shell script
dev_num_list=()
dev_num_list=(`./rvs -g | grep Device |  awk '{ print $6 }' | grep -o -E '[0-9]+'`)
dev_num_str=${dev_num_list[@]}
#echo "dev_num_str $dev_num_str"

function get_string_with_space {
local str=$1 ; shift
local file=$1
local space=" "
str_space=0
str_space=(`grep "$str:" $file | awk -F ":" '{print $1}' | grep "$str" | awk -F'[^ ]' '{print length($1),NR}'`)
echo $str_space
str_space=$(($str_space + 0))
while [ $str_space -gt 0 ]
do
    str=$(printf "%s%s" "$space" "$str")
    str_space=$(($str_space - 1))
    echo $str_space
done
echo $str
ret_str=$str
}

# Below for loop is for going through all .conf files passed as argument to
# deviceid.sh file
for dev_id_line in $1; do
    #echo $dev_id_line
    get_string_with_space "deviceid" $dev_id_line
    configLine  "deviceid" "$ret_str $d_i" $dev_id_line
    get_string_with_space "device" $dev_id_line
    configLine  "device:" "$ret_str: $dev_num_str" $dev_id_line
done
