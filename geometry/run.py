import os
import sys
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--show_cases', '-s', default=False, action="store_true")
    parser.add_argument('--data_root', '-r', type=str, default="../data/olat")
    parser.add_argument('--case_id', '-c', type=int, nargs="+", default=[0])
    parser.add_argument('--key', '-k', type=str, default='')
    parser.add_argument('--device_id', '-d', type=int, default=0)
    parser.add_argument('--all', '-a', default=False, action="store_true")

    args = parser.parse_args(sys.argv[1:])

    case_names = [case_name for case_name in sorted(os.listdir(args.data_root)) if not case_name.endswith('.zip')]
    if args.show_cases:
        print(f"cases: {case_names}")
        exit()

    cid_list = args.case_id
    if args.all:
        cid_list = [i for i in range(len(case_names))]
        print('> run all cases sequentially')

    print('> run cases: {}'.format([case_names[cid] for cid in cid_list]))

    case_cmds = [
        f"CUDA_VISIBLE_DEVICES={args.device_id} python train_neus_olat.py --scene Cat --save_path output/Cat_{args.key} --camopt",
        f"CUDA_VISIBLE_DEVICES={args.device_id} python train_neus_olat.py --scene CatSmall --save_path output/CatSmall_{args.key} --camopt --camopt_lr 3e-6",
        f"CUDA_VISIBLE_DEVICES={args.device_id} python train_neus_olat.py --scene CupFabric --save_path output/CupFabric_{args.key} --camopt",
        f"CUDA_VISIBLE_DEVICES={args.device_id} python train_neus_olat.py --scene Fish --save_path output/Fish_{args.key} --camopt --womask_end 5000",
        f"CUDA_VISIBLE_DEVICES={args.device_id} python train_neus_olat.py --scene FurScene --save_path output/FurScene_{args.key} --camopt --sdf_scale 1.0",
        f"CUDA_VISIBLE_DEVICES={args.device_id} python train_neus_olat.py --scene Pikachu --save_path output/Pikachu_{args.key} --camopt",
        f"CUDA_VISIBLE_DEVICES={args.device_id} python train_neus_olat.py --scene Pixiu --save_path output/Pixiu_{args.key} --camopt",
    ]

    for cid in cid_list:
        
        case_name = case_names[cid]
        run_cmd = case_cmds[cid]

        print(f"> proc case {case_name}")
        print(f"> cmd {run_cmd}")
        os.system(run_cmd)