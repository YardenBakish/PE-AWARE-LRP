from modules.layers_ours import *
from modules.layers_patch_embed import *

from functools import partial

DEFAULT_MODEL = {
    'norm'                 : partial(LayerNorm, eps=1e-6) ,
    'last_norm'            : LayerNorm ,
    'activation'           : GELU(),
    'isWithBias'           : True,
    'isConvWithBias'       : True,

    'attn_activation'      : Softmax(dim=-1),
    'patch_embed'          : PatchEmbed,
    'attn_drop_rate'       : 0.,
    'FFN_drop_rate'        : 0.,
    'projection_drop_rate' : 0.,
    'reg_coeffs'           : None,

}


DEFAULT_PATHS = {

    'imagenet_1k_Dir'        : '/datasets/Imagenet/data/',
    'imagenet_100_Dir'       : '/datasets/Imagenet100/data/',
    'finetuned_models_dir'   : 'finetuned_models/',
    'results_dir'            : 'finetuned_models/',

}





MODEL_VARIANTS = {
            'basic'                          :  DEFAULT_MODEL.copy(),
            'base_small': {**DEFAULT_MODEL, 'size' : 'small'},
            'basic_medium': {**DEFAULT_MODEL, 'size' : 'base'},

}



#chosen randmoly
EPOCHS_TO_PERTURBATE = {
    'IMNET' : {
        'basic':          [0],         
        'base_small': [0],
        'basic_medium': [0],
    }

}


EPOCHS_TO_PERTURBATE_FULL_LRP = {


    'IMNET' : {
        'basic':          [0],           
        'base_small': [0],
        'basic_medium': [0],

    }

}

EPOCHS_TO_SEGMENTATION_FULL_LRP = {
        'basic':          [0],          
        'base_small': [0],
        'basic_medium': [0],

}


EPOCHS_TO_SEGMENTATION = {
        'basic':          [0],          
        'base_small': [0],
        'basic_medium': [0],
}


PRETRAINED_MODELS_URL = {
    'IMNET100': 'finetuned_models/IMNET100/basic/best_checkpoint.pth',
    'IMNET': 'finetuned_models/IMNET/basic/checkpoint_0.pth',
}


XAI_METHODS = ['rollout', 'lrp', 'transformer_attribution', 'attribution_with_detach',
               'full_lrp_semiGammaLinear_alphaConv', 'full_lrp_GammaLinear_alphaConv',  'full_lrp_Linear_alphaConv',
               'full_lrp_semiGammaLinear_gammaConv', 'full_lrp_GammaLinear_gammaConv',  'full_lrp_Linear_gammaConv',
               'lrp_last_layer',
                'attn_last_layer', 'attn_gradcam',
                 'custom_lrp_epsilon_rule',
                'custom_lrp_gamma_rule_default_op', 

                'full_lrp_GammaLinear_POS_ENC_PE_ONLY_gammaConv',

                'custom_lrp_gamma_rule_full', 'custom_lrp_gamma_rule_full_PE_ONLY', 'custom_lrp_gamma_rule_full_SEMANTIC_ONLY',
                'custom_lrp', 'custom_lrp_PE_ONLY',  'custom_lrp_SEMANTIC_ONLY', 

               'full_lrp_semiGammaLinear_POS_ENC_alphaConv', 'full_lrp_GammaLinear_POS_ENC_alphaConv',  'full_lrp_Linear_POS_ENC_alphaConv',
               'full_lrp_semiGammaLinear_POS_ENC_gammaConv', 'full_lrp_GammaLinear_POS_ENC_gammaConv',  'full_lrp_Linear_POS_ENC_gammaConv',

               'full_lrp_semiGammaLinear_POS_GRAD_ENC_alphaConv', 'full_lrp_GammaLinear_POS_GRAD_ENC_alphaConv',  'full_lrp_Linear_POS_GRAD_ENC_alphaConv',
               'full_lrp_semiGammaLinear_POS_GRAD_ENC_gammaConv', 'full_lrp_GammaLinear_POS_GRAD_ENC_gammaConv',  'full_lrp_Linear_POS_GRAD_ENC_gammaConv',
               'full_lrp_semiGammaLinear_POS_GRAD_ENC_gammaConv', 'full_lrp_GammaLinear_POS_GRAD_ENC_gammaConv',  'full_lrp_Linear_POS_GRAD_ENC_gammaConv',

                ]



DEFAULT_GAMMA_LINEAR = 0.05
DEFAULT_GAMMA_CONV = 100
DEFAULT_ALPHA_LINEAR = 1


def set_propogation_rules(args, gridSearch = False):
    args.prop_rules =  {}
    linear_gamma_rule = DEFAULT_GAMMA_LINEAR
    conv_gamma_rule = DEFAULT_GAMMA_CONV
    linear_alpha_rule = 1
    if gridSearch:
        lst = args.method.split("_")
        linear_gamma_rule = float(lst[0]) 
        linear_alpha_rule = float(lst[0])

        conv_gamma_rule   = float(lst[1])



    if ("full_lrp" in args.method):
        args.prop_rules['epsilon_rule'] = True if 'epsilon_rule' in args.method or 'GammaLinear' in args.method else False
        args.prop_rules['conv_gamma_rule']   = conv_gamma_rule if 'gammaConv' in args.method else False
        args.prop_rules['linear_gamma_rule']  = linear_gamma_rule if 'GammaLinear' in args.method else False
        args.prop_rules["linear_alpha_rule"]   = linear_alpha_rule
        args.prop_rules["default_op"]   = True if 'semiGammaLinear' in args.method else False
        args.conv_prop_rule = args.method.split("_")[-1]

    else:
        args.prop_rules['epsilon_rule'] = True if 'epsilon_rule' in args.method or 'gamma_rule' in args.method else False
        args.prop_rules['conv_gamma_rule']    = conv_gamma_rule if 'gamma_rule' in args.method else False
        args.prop_rules['linear_gamma_rule']    = linear_gamma_rule if 'gamma_rule' in args.method else False
        args.prop_rules["linear_alpha_rule"]   = linear_alpha_rule
        args.prop_rules["default_op"]  = True if 'default_op' in args.method else False
        if args.prop_rules["default_op"] == True and args.prop_rules['linear_gamma_rule'] == False:
            print("default_op must come together with gamma_rule")
            exit(1)

    
    args.ext = args.method


def set_components_custom_lrp(args, gridSearch = False):
    if gridSearch == True:
        pass
    else:
        if args.method not in XAI_METHODS:
            print(f"must choose from: {XAI_METHODS}")
            exit(1)
    if args.method == "attribution_with_detach":
        args.epsilon_rule  = False
        args.gamma_rule = False
        args.default_op = False
        args.cp_rule = False
        args.model_components['norm'] = partial(CustomLRPLayerNorm, eps=1e-6)
        args.model_components['last_norm'] = CustomLRPLayerNorm

        if args.variant == "norm_batch":
            args.model_components['norm']      = partial(RepBN, batchLayer = CustomLRPBatchNorm)
            args.model_components['last_norm'] = partial(RepBN, batchLayer = CustomLRPBatchNorm)

        if args.variant == "norm_rms":
            args.model_components['norm'] = partial(CustomLRPRMSNorm, eps=1e-6)
            args.model_components['last_norm'] = CustomLRPRMSNorm

        args.ext = "attribution_w_detach"
    if "custom_lrp" in args.method or "full_lrp" in args.method or "custom_RAP" in args.method:

        set_propogation_rules(args, gridSearch = gridSearch)


        print(f"inside config with custom_lrp")
        if args.variant == "norm_batch":
            args.model_components['norm']      = partial(RepBN, batchLayer = CustomLRPBatchNorm)
            args.model_components['last_norm'] = partial(RepBN, batchLayer = CustomLRPBatchNorm)


            args.cp_rule = True
            return


        args.model_components['norm'] = partial(CustomLRPLayerNorm, eps=1e-6)
        args.model_components['last_norm'] = CustomLRPLayerNorm

        if ('relu' in args.variant) and (args.variant != 'attn_act_relu') and (args.variant != 'act_relu'):
            args.cp_rule = False
            return

        if args.variant == "norm_rms":
            args.model_components['norm'] = partial(CustomLRPRMSNorm, eps=1e-6)
            args.model_components['last_norm'] = CustomLRPRMSNorm



        args.cp_rule = True

    else:
        args.cp_rule = False



def SET_VARIANTS_CONFIG(args):
    if args.variant not in MODEL_VARIANTS:
        print(f"only allowed to use the following variants: {MODEL_VARIANTS.keys()}")
        exit(1)


    args.model_components = MODEL_VARIANTS[args.variant]



def SET_PATH_CONFIG(args):


    if args.data_set == 'IMNET100':
        args.data_path = args.dirs['imagenet_100_Dir']
    else:
        args.data_path = args.dirs['imagenet_1k_Dir']





def get_config(args, skip_further_testing = False, get_epochs_to_perturbate = False, get_epochs_to_segmentation = False):


    SET_VARIANTS_CONFIG(args)

    #if args.custom_lrp:
    #    set_components_custom_lrp(args)
    args.dirs = DEFAULT_PATHS


    if get_epochs_to_perturbate:
        args.epochs_to_perturbate = EPOCHS_TO_PERTURBATE if "full_lrp" not in args.method else EPOCHS_TO_PERTURBATE_FULL_LRP



    if get_epochs_to_segmentation:
        args.epochs_to_segmentation = EPOCHS_TO_SEGMENTATION if "full_lrp" not in args.method else EPOCHS_TO_SEGMENTATION_FULL_LRP
   # if skip_further_testing == False:
   #     vars(args).update(DEFAULT_PARAMETERS)




    if args.data_path == None:
        SET_PATH_CONFIG(args)

    if skip_further_testing:
        return

    if args.auto_start_train:
        args.finetune =  PRETRAINED_MODELS_URL[args.data_set]


    #if args.eval and args.resume =='' and args.auto_resume == False:
    #    print("for evaluation please add --resume  with your model path, or add --auto-resume to automatically find it ")
    #    exit(1)
    if args.verbose:
        print(f"working with model {args.model} | dataset: {args.data_set} | variant: {args.variant}")
