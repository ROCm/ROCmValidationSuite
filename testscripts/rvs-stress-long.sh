if [ "$1" == "-h" ] ; then
        echo
        echo "Usage: ./`basename $0` 3 (for 3 hours) or ./`basename $0` 6 (for 6 hours), default duration is 1 hour"
        echo
    exit 0
fi
if [ "$#" -eq 0 ] ; then
    duration="3600000"
    msval="msval"
    sed -i "s/$msval/${duration}/g" conf/gst_stress.conf
    date
    echo 'gst-1hr';sudo ./rvs -c conf/gst_stress.conf -d 3; date
    sed -i "s/${duration}/$msval/g" conf/gst_stress.conf
else
    duration=$((3600000 * $1))
    echo $duration
    msval="msval"
    sed -i "s/$msval/${duration}/g" conf/gst_stress.conf
    date
    echo 'gst-'$1hr'(s)';sudo ./rvs -c conf/gst_stress.conf -d 3; date
    sed -i "s/${duration}/$msval/g" conf/gst_stress.conf
fi
