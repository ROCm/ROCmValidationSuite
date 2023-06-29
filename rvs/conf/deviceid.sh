# ################################################################################
# #
# # Copyright (c) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
# #
# # MIT LICENSE:
# # Permission is hereby granted, free of charge, to any person obtaining a copy of
# # this software and associated documentation files (the "Software"), to deal in
# # the Software without restriction, including without limitation the rights to
# # use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# # of the Software, and to permit persons to whom the Software is furnished to do
# # so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
# #
# ###############################################################################

#!/bin/bash
# to Use this script need to set RVS_EXE_PATH where your rvs executable is.
# ex. RVS_EXE_PATH=/opt/rocm/rvs/rvs for default rvs installation path
# Run script as below
# Example:  deviceid.sh "conf/gpup_*.conf"<-- conf file path
# Above command will replace device_id/device_id/deviceid : XXXX in all gpup_*.conf files
# To change any single file use as below example
# ./conf/deviceid.sh conf/gpup_1.conf

# setting ROCM_PATH using cmake inputs
# setting RVS_EXE_PATH to default if not set by user
if [[ -z "${RVS_EXE_PATH}" ]]; then
    export RVS_EXE_PATH=/opt/rocm/rvs/rvs
fi

function get_string_with_space {
local str=$1 ; shift
local line=$1; shift
local new_str=$1
#echo $new_str
#echo "After new string"
#printf "%s" "$line"
local space=" "
ret_str=""
str_space=0
str_space=(`printf "%s" "$line" | awk -F ":" '{print $1}' | grep "$str" | awk -F'[^ ]' '{print length($1),NR}'`)
#echo $str_space
str_space=$(($str_space + 0))
while [ $str_space -gt 0 ]
do
    new_str=$(printf "%s%s" "$space" "$new_str")
    str_space=$(($str_space - 1))
#    echo $str_space
done
#echo "before new_str"
#printf "%s" "$new_str"
ret_str=$new_str
}

#Below  configLine function I got from https://stackoverflow.com/a/54909102
# configLine will check for device_id string in current file and replace or
# add device_id if not present
function configLine {
    local OLD_LINE_PATTERN=$1; shift
    local NEW_LINE=$1; shift
    local FILE=$1
    local line_num=1
    while IFS= read -r line
    do
 #       printf "%s" $line > file.txt
        
        if printf "%s" "$line" | awk -F ":" '{print $1}' | grep "\b$OLD_LINE_PATTERN\b" ; then
            if  [[ $line !=  *"#"* ]] ; then 
                #echo ${OLD_LINE_PATTERN} 
                #echo "$line"
 #               echo $NEW_LINE
 #               echo "before get_string_with_space"
                get_string_with_space "$OLD_LINE_PATTERN" "$line" "$NEW_LINE"
                sed -i "${line_num}s/.*/${ret_str}/" "${FILE}"
            fi
        fi
        let "line_num=line_num+1"
    done < "$FILE"
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
if [[ -z "${RVS_EXE_PATH}" ]]; then
  echo "ERROR: RVS_EXE_PATH not set, First set RVS_EXE_PATH environment path"
  exit 0
else
  rvs_exe_path="${RVS_EXE_PATH}"
fi
dev_num_list=(`$rvs_exe_path -g | grep Device |  awk '{ print $6 }' | grep -o -E '[0-9]+'`)
dev_num_str=${dev_num_list[@]}
#echo "dev_num_str $dev_num_str"

# Below for loop is for going through all .conf files passed as argument to
# deviceid.sh file
for dev_id_line in $1; do
    configLine  "device_id" "device_id: $d_i" $dev_id_line
    configLine  "deviceid" "deviceid: $d_i" $dev_id_line
    configLine  "peer_deviceid" "peer_deviceid: $d_i" $dev_id_line
    configLine  "peer_device_id" "peer_device_id: $d_i" $dev_id_line
    configLine  "device" "device: $dev_num_str" $dev_id_line
    configLine  "peers" "peers: $dev_num_str" $dev_id_line
done
