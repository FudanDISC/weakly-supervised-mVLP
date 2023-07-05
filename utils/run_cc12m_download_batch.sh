dir_name=$(dirname $0)
echo $dir_name
mkdir /root/tmp_imgs/
mkdir /root/tmp_imgs/cc12m_sub/
for ((i=$1; i<$2; i++));
do
s_start=$(expr $i \* $3)
s_end=$(expr $i \* $3 + $3)
s_end=$((s_end > $4 ? $4 : s_end))
echo $s_start
echo $s_end
python3 $dir_name/tsv_download.py --tsv_file /remote-home/zjli/tmp_data/cc12m.tsv --img_dir /root/tmp_imgs/cc12m_sub/ --start_id $s_start --end_id $s_end --thread_num 100  --url_index 0
done