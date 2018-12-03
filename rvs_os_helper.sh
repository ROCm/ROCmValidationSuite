#!/bin/bash
cat /etc/os-release|grep ^ID=|sed 's/ID=//;s/"//g'