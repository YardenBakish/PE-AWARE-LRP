import os

import argparse
import subprocess
import config



def parse_args():
    parser = argparse.ArgumentParser(description='evaluate perturbations')
    parser.add_argument('--pass-vis', action='store_true', help= "skip generating visualizations")
    parser.add_argument('--normalized-pert', type=int, default=1, choices = [0,1])

    parser.add_argument('--fract', type=float,
                        default=0.1,
                        help='')

    parser.add_argument('--grid', action='store_true')
    
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    
    parser.add_argument('--work-env', type=str,
                        required= True,
                        help='')
    

        
    parser.add_argument('--variant', default = 'basic', help="")
    
    parser.add_argument('--output-dir', type=str,
                        help='')
    parser.add_argument('--neg', type=int, choices = [0,1], default = 0)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--custom-trained-model', type=str,help='')
    parser.add_argument('--data-set', default='IMNET100', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--num-workers', type=int,
                        default= 1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                     
                        help='')

    parser.add_argument('--both',  action='store_true')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Runs the first 5 samples and visualizes ommited pixels')
    parser.add_argument('--wrong', action='store_true',
                        default=False,
                        help='')

    parser.add_argument('--scale', type=str,
                        default='per',
                        choices=['per', '100'],
                        help='')

    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.', default=True)
    parser.add_argument('--lmd', type=float,
                        default=10,
                        help='')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='')
    parser.add_argument('--cls-agn', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-ia', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fgx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-m', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-reg', action='store_true',
                        default=False,
                        help='')

    parser.add_argument('--data-path', type=str,
                   
                        help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args                   = parse_args()

    config.get_config(args, skip_further_testing = True)

    if 'work_env' not in args.work_env:
       print("work_env term must be included in your --work-env arg")
       exit(1)

    run_gen_vis_cmd        = "python generate_visualizations.py"
    run_gen_pert_cmd       = "python generate_perturbations.py"
    
    run_gen_vis_cmd       +=  f' --method {args.method}'
    run_gen_pert_cmd      +=  f' --method {args.method}'
 
    run_gen_vis_cmd       +=  f' --data-path {args.data_path}'

    run_gen_vis_cmd       +=  f' --data-set {args.data_set}'
    run_gen_pert_cmd      +=  f' --data-set {args.data_set}'

    run_gen_vis_cmd       +=  f' --batch-size {args.batch_size}'
    run_gen_pert_cmd      +=  f' --batch-size {args.batch_size}'

    run_gen_vis_cmd       +=  f' --num-workers {args.num_workers}'
    
   
    run_gen_vis_cmd       +=  f' --variant {args.variant}'
    run_gen_pert_cmd      +=  f' --variant {args.variant}'
   
    run_gen_vis_cmd       +=  f' --custom-trained-model {args.custom_trained_model}'
    run_gen_pert_cmd      +=  f' --custom-trained-model {args.custom_trained_model}'


    run_gen_vis_cmd       +=  f' --work-env {args.work_env}'
    run_gen_pert_cmd      +=  f'  --work-env {args.work_env}'

    run_gen_vis_cmd       +=  f' --normalized-pert {args.normalized_pert}'
    run_gen_pert_cmd      +=  f' --normalized-pert {args.normalized_pert}'

    run_gen_vis_cmd       +=  f' --fract {args.fract}'

    if args.grid:
       run_gen_vis_cmd       +=  f' --grid'
       run_gen_pert_cmd       +=  f' --grid'

    
    
    os.makedirs(args.work_env, exist_ok=True)
    
    if args.output_dir:
      os.makedirs(args.output_dir, exist_ok=True)
      run_gen_pert_cmd +=  f' --output-dir {args.output_dir}'


       
    if args.pass_vis == False:
      try:
        subprocess.run(run_gen_vis_cmd, check=True, shell=True)
        print(f"generated visualizations")
      except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)
    
    run_twice              = False
    run_gen_pert_cmd_opp = None
    if args.both:
        run_twice = True
        run_gen_pert_cmd_opp = run_gen_pert_cmd +  f' --neg {1-(args.neg)}'

    else:
        run_gen_pert_cmd +=  f' --neg {args.neg}'
       
    try:
      subprocess.run(run_gen_pert_cmd, check=True, shell=True)
      print(f"generated visualizations")
    except subprocess.CalledProcessError as e:
      print(f"Error: {e}")
      exit(1)
    
    print("starting again")
    if run_twice:
        try:
          subprocess.run(run_gen_pert_cmd_opp, check=True, shell=True)
          print(f"generated visualizations")
        except subprocess.CalledProcessError as e:
          print(f"Error: {e}")
          exit(1)

    if args.work_env:
      subprocess.run(f'rm -rf {args.work_env}',check=True, shell=True)
       
    