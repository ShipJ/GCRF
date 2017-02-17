
loc_code = 'sen'
clust_type = 'hex'
size = 1000

python bts_aggregate.py $loc_code $clust_type $size
python radiation.py $loc_code $clust_type $size
python gravity.py $loc_code $clust_type $size

