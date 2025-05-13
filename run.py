import os
import sys
import glob
from argparse import ArgumentParser


if __name__ == '__main__':
    # run train, render and metric in a single script
    # add args, keyword, port, device ids
    parser = ArgumentParser()
    parser.add_argument('--show_cases', '-s', default=False, action="store_true")
    parser.add_argument('--port', '-p', type=int, default=6009)
    parser.add_argument('--root_folder', '-r', type=str, default='data/olat')
    parser.add_argument('--case_id', '-c', type=int, nargs='+', default=[0])
    parser.add_argument('--key', '-k', type=str, default='')
    parser.add_argument('--device_id', '-d', type=int, default=0)
    parser.add_argument('--all', '-a', default=False, action="store_true")

    args = parser.parse_args(sys.argv[1:])

    case_names = [case_name for case_name in sorted(os.listdir(args.root_folder)) if not case_name.endswith('.zip')]

    if args.show_cases:
        print('cases: {}'.format(case_names))
        exit()

    cid_list = args.case_id
    if args.all:
        cid_list = [i for i in range(len(case_names))]
        print('> run all cases sequentially')

    print('> run cases: {}'.format([case_names[cid] for cid in cid_list]))

    for cid in cid_list:

        case_name = case_names[cid]
        data_folder = 'data'
        case_folder = os.path.join(data_folder, case_name)

        train_cmd = "CUDA_VISIBLE_DEVICES={} python train.py -s {} --eval --port {} --key {}"\
            .format(args.device_id, os.path.join(args.root_folder, case_name), args.port, args.key)

        os.system(train_cmd)

        # get result with key
        case_results_list = os.listdir(os.path.join('output', case_name))
        result_tag = None
        for result in case_results_list:
            if args.key == result[9:]:
                result_tag = result
                break
        if result_tag is None:
            print('[not found] trained result key {}'.format(args.key))
            exit()

        result_fdr = os.path.join('output', case_name, result_tag)
        render_cmd = "CUDA_VISIBLE_DEVICES={} python render.py -m {} --skip_train".format(args.device_id, result_fdr)

        os.system(render_cmd)

        os.system("CUDA_VISIBLE_DEVICES={} python metrics.py -m {}".format(args.device_id, result_fdr))
