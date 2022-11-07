# noise-free degradations with isotropic Gaussian blurs
python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set5' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=563 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="0.8,0.914,1.03,1.14,1.26,1.37,1.48,1.6" \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set14' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=563 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="0.8,0.914,1.03,1.14,1.26,1.37,1.48,1.6" \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='B100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=563 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="0.8,0.914,1.03,1.14,1.26,1.37,1.48,1.6" \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Urban100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=563 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="0.8,0.914,1.03,1.14,1.26,1.37,1.48,1.6" \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set5' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='3' \
               --resume=584 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="1.35,1.5,1.65,1.8,1.95,2.1,2.25,2.40" \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set14' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='3' \
               --resume=584 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="1.35,1.5,1.65,1.8,1.95,2.1,2.25,2.40" \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='B100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='3' \
               --resume=584 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="1.35,1.5,1.65,1.8,1.95,2.1,2.25,2.40" \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Urban100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='3' \
               --resume=584 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="1.35,1.5,1.65,1.8,1.95,2.1,2.25,2.40" \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set5' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=473 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="1.8,2,2.2,2.4,2.6,2.8,3.0,3.2" \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set14' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=473 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="1.8,2,2.2,2.4,2.6,2.8,3.0,3.2" \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='B100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=473 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="1.8,2,2.2,2.4,2.6,2.8,3.0,3.2" \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Urban100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=473 \
               --blur_type='iso_gaussian' \
               --noise_test=0.0 \
               --sig="1.8,2,2.2,2.4,2.6,2.8,3.0,3.2" \
               --n_GPUs=0