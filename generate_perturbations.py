
import torch
import os
from tqdm import tqdm
import numpy as np
import argparse
#from dataset.label_index_corrector import *
from misc.helper_functions import *
from sklearn.metrics import auc
from torchvision.transforms import  GaussianBlur
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms, datasets
from old.model_ablation import deit_tiny_patch16_224 as vit_LRP

#from models.model_wrapper import model_env 
from models.model_handler import model_env 

import glob

from dataset.expl_hdf5 import ImagenetResults
import config


DEBUG_MAX_ITER = 3


  

imagenet_normalize = transforms.Compose([
    #transforms.Resize(256, interpolation=3),
    #transforms.CenterCrop(224),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

def calc_auc(perturbation_steps,matt,op):
    means = []
    for row in matt:
        non_negative_values = row[row >= -10000000]
        if non_negative_values.size > 0:
            row_mean = np.mean(non_negative_values)
        else:
            row_mean = np.nan  
        
        means.append(row_mean)
    auc_score = auc(perturbation_steps, means)
    print(f"\n {op}: AUC: {auc_score}  | means: {means} \n")
    return {f'{exp_name}_{op}': means, f'{exp_name}_auc_{op}':auc_score} 

def eval(args, mode = None):
    
    num_samples          = 0
    num_correct_model    = np.zeros((len(imagenet_ds,)))
    dissimilarity_model  = np.zeros((len(imagenet_ds,)))
    model_index          = 0

    if args.scale == 'per':
        base_size = 224 * 224
        perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    elif args.scale == '100':
        base_size = 100
        perturbation_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    else:
        raise Exception('scale not valid')

    correctence_target_precentage          = np.full((9,len(imagenet_ds)),-1)
    correctence_top_precentage             = np.full((9,len(imagenet_ds)),-1)
    correctence_target_precentage_blurred  = np.full((9,len(imagenet_ds)),-1)
    correctence_top_precentage_blurred     = np.full((9,len(imagenet_ds)),-1)
    correctence_target_precentage_averaged = np.full((9,len(imagenet_ds)),-1)
    correctence_top_precentage_averaged    = np.full((9,len(imagenet_ds)),-1)

    logit_pertubed_ARR                  = np.full((9,len(imagenet_ds)),-1,dtype=np.float64)
    logit_pertubed_blurred_ARR          = np.full((9,len(imagenet_ds)),-1,dtype=np.float64)
    logit_pertubed_averaged_ARR         = np.full((9,len(imagenet_ds)),-1,dtype=np.float64)
    probability_pertubed_ARR            = np.full((9,len(imagenet_ds)),-1,dtype=np.float64)
    probability_pertubed_blurred_ARR    = np.full((9,len(imagenet_ds)),-1,dtype=np.float64)
    probability_pertubed_averaged_ARR   = np.full((9,len(imagenet_ds)),-1,dtype=np.float64)


    num_correct_pertub       = np.zeros((9, len(imagenet_ds)))
    dissimilarity_pertub     = np.zeros((9, len(imagenet_ds)))
    logit_diff_pertub        = np.zeros((9, len(imagenet_ds)))
    prob_diff_pertub         = np.zeros((9, len(imagenet_ds)))
    perturb_index            = 0
    iter_count               = 0
   
    last_label               = None
    blur_transform = GaussianBlur(kernel_size=(51, 51), sigma=(20, 20))
 
    for batch_idx, (data, vis_pred, vis_target, target) in enumerate(tqdm(sample_loader)):
        


        #print(f"\n\n REAL TARGET : {target}")
        if args.debug :
          if last_label == None or last_label != target:
              last_label   = target
              iter_count  +=1
          else:
              continue
          if iter_count > DEBUG_MAX_ITER:
              break
      

        num_samples += len(data)
        data         = data.to(device)
        vis_pred     = vis_pred.to(device)
        vis_target   = vis_target.to(device)
        target       = target.to(device)
        norm_data    = imagenet_normalize(data.clone())

        # Compute model accuracy
        pred               = model(norm_data).detach()
        pred_probabilities = torch.softmax(pred, dim=1)
        pred_org_logit     = pred.data.max(1, keepdim=True)[0].squeeze(1)
        pred_org_prob      = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)
        pred_class         = pred.data.max(1, keepdim=True)[1].squeeze(1)
        
        
        tgt_pred           = (target == pred_class).type(target.type()).data.cpu().numpy()
        num_correct_model[model_index:model_index+len(tgt_pred)] = tgt_pred

        probs        = torch.softmax(pred, dim=1)
        target_probs = torch.gather(probs, 1, target[:, None])[:, 0]
        second_probs = probs.data.topk(2, dim=1)[0][:, 1]
        temp         = torch.log(target_probs / second_probs).data.cpu().numpy()
        dissimilarity_model[model_index:model_index+len(temp)] = temp

        if args.wrong:
            exit(1)
            #wid = np.argwhere(tgt_pred == 0).flatten()
            #if len(wid) == 0:
            #    continue
            #wid = torch.from_numpy(wid).to(vis.device)
            #vis = vis.index_select(0, wid)
            #data = data.index_select(0, wid)
            #target = target.index_select(0, wid)

        # Save original shape
        org_shape = data.shape

        if args.neg:
            vis_pred = -vis_pred
            vis_target = -vis_target


        vis_pred   = vis_pred.reshape(org_shape[0], -1)
        vis_target = vis_target.reshape(org_shape[0], -1)



        for i in range(len(perturbation_steps)):
            _data_pred_pertubarted           = data.clone()
            _data_target_pertubarted         = data.clone()
            _data_blurred_pred_pertubarted   = data.clone()
            _data_blurred_target_pertubarted = data.clone()
            _data_average_pred_pertubarted   = data.clone()
            _data_average_target_pertubarted = data.clone()

            blurred_data = blur_transform(_data_blurred_pred_pertubarted)

            _, idx_pred   = torch.topk(vis_pred, int(base_size * perturbation_steps[i]), dim=-1)
            _, idx_target = torch.topk(vis_target, int(base_size * perturbation_steps[i]), dim=-1)

            idx_pred   = idx_pred.unsqueeze(1).repeat(1, org_shape[1], 1)
            idx_target = idx_target.unsqueeze(1).repeat(1, org_shape[1], 1)



            _data_pred_pertubarted           = _data_pred_pertubarted.reshape(org_shape[0], org_shape[1], -1)
            _data_target_pertubarted         = _data_target_pertubarted.reshape(org_shape[0], org_shape[1], -1)
            _data_blurred_pred_pertubarted   = _data_blurred_pred_pertubarted.reshape(org_shape[0], org_shape[1], -1)
            _data_blurred_target_pertubarted = _data_blurred_target_pertubarted.reshape(org_shape[0], org_shape[1], -1)
            _data_average_pred_pertubarted   = _data_average_pred_pertubarted.reshape(org_shape[0], org_shape[1], -1)
            _data_average_target_pertubarted = _data_average_target_pertubarted.reshape(org_shape[0], org_shape[1], -1)
            
            #for mean
            mean_pixel0 = _data_average_pred_pertubarted.mean(dim=-1,keepdim = True) 
            mean_pixel0 =mean_pixel0.expand(-1, -1, idx_pred.size(-1)) 
          


            _data_pred_pertubarted   = _data_pred_pertubarted.scatter_(-1, idx_pred, 0)
            _data_target_pertubarted = _data_target_pertubarted.scatter_(-1, idx_target, 0)

            _data_blurred_pred_pertubarted = _data_blurred_pred_pertubarted.scatter_(-1, idx_pred, blurred_data.reshape(org_shape[0], org_shape[1], -1).gather(-1, idx_pred))            
            _data_blurred_target_pertubarted = _data_blurred_target_pertubarted.scatter_(-1, idx_target, blurred_data.reshape(org_shape[0], org_shape[1], -1).gather(-1, idx_target))

            _data_average_pred_pertubarted   = _data_average_pred_pertubarted.scatter_(-1, idx_pred, mean_pixel0 )
            _data_average_target_pertubarted = _data_average_target_pertubarted.scatter_(-1, idx_target, mean_pixel0 )





            _data_pred_pertubarted               = _data_pred_pertubarted.reshape(*org_shape)
            _data_target_pertubarted             = _data_target_pertubarted.reshape(*org_shape)
            _data_blurred_pred_pertubarted       = _data_blurred_pred_pertubarted.reshape(*org_shape)
            _data_blurred_target_pertubarted     = _data_blurred_target_pertubarted.reshape(*org_shape)

            _data_average_pred_pertubarted       = _data_average_pred_pertubarted.reshape(*org_shape)
            _data_average_target_pertubarted     = _data_average_target_pertubarted.reshape(*org_shape)
          

       

            #dbueg
            if args.debug:
                os.makedirs(f'testing/pert_vis/{batch_idx}', exist_ok=True)
                np.save(f"testing/pert_vis/{batch_idx}/pert_{i}",  _data_pred_pertubarted.cpu().numpy())
                np.save(f"testing/pert_vis/{batch_idx}/pert_black{i}",  _data_blurred_pred_pertubarted.cpu().numpy())   

            
            _norm_data_pred_pertubarted            = imagenet_normalize(_data_pred_pertubarted)
            _norm_data_target_pertubarted          = imagenet_normalize(_data_target_pertubarted)
            _norm_data_blurred_pred_pertubarted    = imagenet_normalize(_data_blurred_pred_pertubarted)
            _norm_data_blurred_target_pertubarted  = imagenet_normalize(_data_blurred_target_pertubarted)
            _norm_data_average_pred_pertubarted    = imagenet_normalize(_data_average_pred_pertubarted)
            _norm_data_average_target_pertubarted  = imagenet_normalize(_data_average_target_pertubarted)
            
            out_data_pred_pertubarted           = model(_norm_data_pred_pertubarted).detach()
            out_data_target_pertubarted         = model(_norm_data_target_pertubarted).detach()
            out_data_blurred_pred_pertubarted   = model(_norm_data_blurred_pred_pertubarted).detach()
            out_data_blurred_target_pertubarted = model(_norm_data_blurred_target_pertubarted).detach()
            out_data_average_pred_pertubarted   = model(_norm_data_average_pred_pertubarted).detach()
            out_data_average_target_pertubarted = model(_norm_data_average_target_pertubarted).detach()

            #print(pred_class.shape)
            #print(out_data_pred_pertubarted.shape)

            logit_pertubed_predicted            = out_data_pred_pertubarted[torch.arange(out_data_pred_pertubarted.size(0)),pred_class].data.cpu().numpy()
            #print(logit_pertubed_predicted.shape)
            logit_pertubed_predicted_blurred    = out_data_blurred_pred_pertubarted[torch.arange(out_data_pred_pertubarted.size(0)),pred_class].data.cpu().numpy()
            logit_pertubed_predicted_average    = out_data_average_pred_pertubarted[torch.arange(out_data_pred_pertubarted.size(0)),pred_class].data.cpu().numpy()



            pred_probabilities         = torch.softmax(out_data_pred_pertubarted, dim=1)
            pred_probabilities_blurred = torch.softmax(out_data_blurred_pred_pertubarted, dim=1)
            pred_probabilities_average = torch.softmax(out_data_average_pred_pertubarted, dim=1)


            prob_pertubed_predicted         = pred_probabilities[torch.arange(out_data_pred_pertubarted.size(0)),pred_class].data.cpu().numpy()
            prob_pertubed_predicted_blurred = pred_probabilities_blurred[torch.arange(out_data_pred_pertubarted.size(0)),pred_class].data.cpu().numpy()
            prob_pertubed_predicted_average = pred_probabilities_average[torch.arange(out_data_pred_pertubarted.size(0)),pred_class].data.cpu().numpy()



            pred_prob = pred_probabilities.data.max(1, keepdim=True)[0].squeeze(1)
            
            pred_class_pertubtated         = out_data_pred_pertubarted.data.max(1, keepdim=True)[1].squeeze(1)
            pred_class_pertubtated_blur    = out_data_blurred_pred_pertubarted.data.max(1, keepdim=True)[1].squeeze(1)
            pred_class_pertubtated_average = out_data_average_pred_pertubarted.data.max(1, keepdim=True)[1].squeeze(1)




            diff = (pred_prob - pred_org_prob).data.cpu().numpy()
            prob_diff_pertub[i, perturb_index:perturb_index+len(diff)] = diff

            pred_logit = out_data_pred_pertubarted.data.max(1, keepdim=True)[0].squeeze(1)
            diff = (pred_logit - pred_org_logit).data.cpu().numpy()
            logit_diff_pertub[i, perturb_index:perturb_index+len(diff)] = diff

            target_class         = out_data_target_pertubarted.data.max(1, keepdim=True)[1].squeeze(1)
            target_class_blurred = out_data_blurred_target_pertubarted.data.max(1, keepdim=True)[1].squeeze(1)
            target_class_average = out_data_average_target_pertubarted.data.max(1, keepdim=True)[1].squeeze(1)



            temp        = (target == target_class).type(target.type()).data.cpu().numpy()
            tempBlurred = (target == target_class_blurred).type(target.type()).data.cpu().numpy()
            tempAverage = (target == target_class_average).type(target.type()).data.cpu().numpy()



            isCorrectOnInitPred        = (pred_class == pred_class_pertubtated).type(target.type()).data.cpu().numpy()
            isCorrectOnInitPredBlurred = (pred_class == pred_class_pertubtated_blur).type(target.type()).data.cpu().numpy()
            isCorrectOnInitPredAverage = (pred_class == pred_class_pertubtated_average).type(target.type()).data.cpu().numpy()

            


            isCorrect =temp
            isCorrectBlurred =tempBlurred
            isCorrectAverage =tempAverage



            num_correct_pertub[i, perturb_index:perturb_index+len(temp)] = temp

            probs_pertub = torch.softmax(out_data_pred_pertubarted, dim=1)
            target_probs = torch.gather(probs_pertub, 1, target[:, None])[:, 0]
            second_probs = probs_pertub.data.topk(2, dim=1)[0][:, 1]
            temp = torch.log(target_probs / second_probs).data.cpu().numpy()
            dissimilarity_pertub[i, perturb_index:perturb_index+len(temp)] = temp
            #print(i,batch_idx)
            correctence_top_precentage[i, perturb_index:perturb_index+len(temp)]               = isCorrectOnInitPred
            correctence_top_precentage_blurred[i, perturb_index:perturb_index+len(temp)]       = isCorrectOnInitPredBlurred
            correctence_top_precentage_averaged[i, perturb_index:perturb_index+len(temp)]      = isCorrectOnInitPredAverage


            correctence_target_precentage[i, perturb_index:perturb_index+len(temp)]            = isCorrect
            correctence_target_precentage_blurred[i, perturb_index:perturb_index+len(temp)]    = isCorrectBlurred
            correctence_target_precentage_averaged[i, perturb_index:perturb_index+len(temp)]   = isCorrectAverage


            logit_pertubed_ARR[i,perturb_index:perturb_index+len(temp)]       = logit_pertubed_predicted
            probability_pertubed_ARR[i,perturb_index:perturb_index+len(temp)] = prob_pertubed_predicted

            logit_pertubed_blurred_ARR[i,perturb_index:perturb_index+len(temp)] =  logit_pertubed_predicted_blurred
            probability_pertubed_blurred_ARR[i,perturb_index:perturb_index+len(temp)] = prob_pertubed_predicted_blurred

            
            logit_pertubed_averaged_ARR[i,perturb_index:perturb_index+len(temp)] = logit_pertubed_predicted_average
            probability_pertubed_averaged_ARR[i,perturb_index:perturb_index+len(temp)] = prob_pertubed_predicted_average


        model_index += len(target)
        perturb_index += len(target)
        
    # np.save(os.path.join(args.experiment_dir, 'model_hits.npy'), num_correct_model)
    # np.save(os.path.join(args.experiment_dir, 'model_dissimilarities.npy'), dissimilarity_model)
    # np.save(os.path.join(args.experiment_dir, 'perturbations_hits.npy'), num_correct_pertub[:, :perturb_index])
    # np.save(os.path.join(args.experiment_dir, 'perturbations_dissimilarities.npy'), dissimilarity_pertub[:, :perturb_index])
    # np.save(os.path.join(args.experiment_dir, 'perturbations_logit_diff.npy'), logit_diff_pertub[:, :perturb_index])
    # np.save(os.path.join(args.experiment_dir, 'perturbations_prob_diff.npy'), prob_diff_pertub[:, :perturb_index])
    
    #print(correctence_target_precentage)

    op1 = "target"
    op2 = "top"

    final_res = {}
        
    res_target       = calc_auc(perturbation_steps,correctence_target_precentage,op1)
    res_top          = calc_auc(perturbation_steps,correctence_top_precentage,op2)  
    
    res_target_blur  = calc_auc(perturbation_steps,correctence_target_precentage_blurred, f'{op1}_blur')
    res_top_blur     = calc_auc(perturbation_steps,correctence_top_precentage_blurred,f'{op2}_blur')

    res_target_avg   = calc_auc(perturbation_steps,correctence_target_precentage_averaged, f'{op1}_average')
    res_top_avg      = calc_auc(perturbation_steps,correctence_top_precentage_averaged,f'{op2}_average')
    
    final_res_binary = {**res_target, **res_top,**res_target_blur,**res_top_blur,**res_target_avg,**res_top_avg}
    logits           = calc_auc(perturbation_steps,logit_pertubed_ARR,"logits")
    confidece        = calc_auc(perturbation_steps,probability_pertubed_ARR,"probalities")

    logits_blur           = calc_auc(perturbation_steps,logit_pertubed_blurred_ARR,"logits_blur")
    confidece_blur        = calc_auc(perturbation_steps,probability_pertubed_blurred_ARR,"probalities_blur")

    logits_average          = calc_auc(perturbation_steps,logit_pertubed_averaged_ARR,"logits_average")
    confidece_average        = calc_auc(perturbation_steps,probability_pertubed_averaged_ARR,"probalities_average")


    final_res_logits = {**logits, **confidece,**logits_blur,**confidece_blur,**logits_average,**confidece_average}

    final_res = {**final_res_binary, **final_res_logits}

    if args.output_dir:
        update_json(f"{args.output_dir}/pert_results.json", final_res)

    
   
    #print(np.mean(num_correct_model), np.std(num_correct_model))
    #print(np.mean(dissimilarity_model), np.std(dissimilarity_model))
    #print(perturbation_steps)
    #print(np.mean(num_correct_pertub, axis=1), np.std(num_correct_pertub, axis=1))
    #print(np.mean(dissimilarity_pertub, axis=1), np.std(dissimilarity_pertub, axis=1))







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    
    parser.add_argument('--normalized-pert', type=int,
                        default=1,
                        choices = [0,1])

    parser.add_argument('--work-env', type=str,
                        required = True,
                        help='')
    
    parser.add_argument('--output-dir', type=str,
                        help='')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--custom-trained-model', type=str,help='')
    

    parser.add_argument('--variant', default = 'basic', help="")
    
    parser.add_argument('--data-set', default='IMNET100', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.', default=True)
    parser.add_argument('--neg', type=int, choices = [0,1], default = 0)
    parser.add_argument('--debug', 
                    
                        action='store_true',
                        help='Runs the first 5 samples and visualizes ommited pixels')
    parser.add_argument('--scale', type=str,
                        default='per',
                        choices=['per', '100'],
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                        help='')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='')
    parser.add_argument('--wrong', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='')
    parser.add_argument('--is-ablation', type=bool,
                        default=False,
                        help='')
    

    parser.add_argument('--data-path', type=str, help='')
    parser.add_argument('--grid', action='store_true')

    args = parser.parse_args()

   
    config.get_config(args, skip_further_testing = True)
    config.set_components_custom_lrp(args, gridSearch= args.grid)

    torch.multiprocessing.set_start_method('spawn')

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    if args.work_env:
        PATH = args.work_env
 
    os.makedirs(os.path.join(PATH, 'experiments'), exist_ok=True)
    os.makedirs(os.path.join(PATH, 'experiments/perturbations'), exist_ok=True)

    exp_name  = args.method
    exp_name += '_neg' if args.neg else '_pos'
    print(f"Starting Experiment:{exp_name}")

    if args.vis_class == 'index':
        args.runs_dir = os.path.join(PATH, 'experiments/perturbations/{}/{}_{}'.format(exp_name,
                                                                                       args.vis_class,
                                                                                       args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        args.runs_dir = os.path.join(PATH, 'experiments/perturbations/{}/{}/{}'.format(exp_name,
                                                                                    args.vis_class, ablation_fold))
        # args.runs_dir = os.path.join(PATH, 'experiments/perturbations/{}/{}'.format(exp_name,
        #                                                                             args.vis_class))

    if args.wrong:
        args.runs_dir += '_wrong'

    experiments         = sorted(glob.glob(os.path.join(args.runs_dir, 'experiment_*')))
    experiment_id       = int(experiments[-1].split('_')[-1]) + 1 if experiments else 0
    args.experiment_dir = os.path.join(args.runs_dir, 'experiment_{}'.format(str(experiment_id)))
    os.makedirs(args.experiment_dir, exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if args.vis_class == 'index':
        vis_method_dir = os.path.join(PATH,'visualizations/{}/{}_{}'.format(args.method,
                                                          args.vis_class,
                                                          args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        vis_method_dir = os.path.join(PATH,'visualizations/{}/{}/{}'.format(args.method,
                                                       args.vis_class, ablation_fold))


    imagenet_ds = ImagenetResults(vis_method_dir)


    # Model

    if args.custom_trained_model != None:
        if args.data_set == 'IMNET100':
            args.nb_classes = 100
        else:
            args.nb_classes = 1000

        model = model_env(pretrained=False, 
                      args = args,
                      hooks = True,
                    )
        #model_LRP.head = torch.nn.Linear(model.head.weight.shape[1],100)
        checkpoint = torch.load(args.custom_trained_model, map_location='cpu')

        model.load_state_dict(checkpoint['model'], strict=False)
        model.to(device)
    else:
        #FIXME: currently only attribution method is tested. Add support for other methods using other variants 
        model = vit_LRP(pretrained=True).cuda()
  
    model.eval()

    save_path = PATH + 'results/'

    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        num_workers=1,
        drop_last = False,
        shuffle=False)

    eval(args)
    
    #if we do not use normalized pert we run a second run for blur
   # if args.normalized_pert == 0:
   #     eval(args, "blur")

