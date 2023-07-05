dir_name=$(dirname $0)
echo $dir_name
for ((i=$1; i<$2; i++));
do
s_start=$(expr $i \* $3)
s_end=$(expr $i \* $3 + $3)
s_end=$((s_end > $4 ? $4 : s_end))
echo $s_start
echo $s_end
python3 $dir_name/tsv_download.py --tsv_file Train_GCC-training.tsv --img_dir images/ --start_id $s_start --end_id $s_end --thread_num 100  --url_index 1
done