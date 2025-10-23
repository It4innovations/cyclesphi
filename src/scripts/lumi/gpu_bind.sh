#!/bin/bash

#source /opt/omnitrace/share/omnitrace/setup-env.sh

export HIP_VISIBLE_DEVICES=$PMI_LOCAL_RANK
#echo $HIP_VISIBLE_DEVICES
#echo $HIP_VISIBLE_DEVICES
$@