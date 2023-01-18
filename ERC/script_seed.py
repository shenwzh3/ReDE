
import os
import numpy as np
os.environ['MKL_THREADING_LAYER'] = 'GNU'



if __name__ == '__main__':
    seed_list = np.random.randint(0, 1000, 8)
    dataset = 'MELD'
    for seed in seed_list:
        cmd = 'python -m torch.distributed.launch --nproc_per_node 4 run.py --dataset MELD --lr 1e-05 --grad_accumulate_step 1 --max_sent_len 200 --batch_size 4 --max_hop 7' + ' --seed ' + str(seed)

        with open('results.txt', 'a') as f:
            f.write(cmd + '\n')

        os.system(cmd)

        print('----------------------------------------------')
