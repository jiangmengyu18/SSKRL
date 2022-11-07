# noise-free degradations with isotropic Gaussian blurs
python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set5' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=600 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=11 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=0 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set14' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=600 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=11 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=0 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='B100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=600 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=11 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=0 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Urban100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=600 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=11 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=0 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set5' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=600 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=11 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=15\
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set14' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=600 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=11 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=15 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='B100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=600 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=11 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=15 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Urban100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='2' \
               --resume=600 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=11 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=15 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set5' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=591 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=21 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=0 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set14' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=591 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=21 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=0 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='B100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=591 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=21 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=0 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Urban100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=591 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=21 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=0 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set5' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=591 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=21 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=15\
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Set14' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=591 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=21 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=15 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='B100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=591 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=21 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=15 \
               --n_GPUs=0

python test.py --test_only \
               --dir_data='/home/xingxm/Datasets' \
               --data_test='Urban100' \
               --model='blindsr' \
               --mode='s-fold' \
               --scale='4' \
               --resume=591 \
               --blur_type='aniso_gaussian' \
               --blur_kernel=21 \
               --theta="0,45,90,135,0,45,90,135" \
               --lambda_1="0.8,0.8,0.8,0.8,2.0,2.0,2.0,2.0" \
               --lambda_2="1.6,1.6,1.6,1.6,4.0,4.0,4.0,4.0" \
               --noise_test=15 \
               --n_GPUs=0