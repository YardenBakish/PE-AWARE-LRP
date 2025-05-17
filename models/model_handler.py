from models.model import deit_tiny_patch16_224 as vit_LRP
from models.model import deit_base_patch16_224 as vit_LRP_base
from models.model import deit_small_patch16_224 as vit_LRP_small

from models.model_train import deit_tiny_patch16_224 as vit_LRP_train
from models.model_train import deit_base_patch16_224 as vit_LRP_base_train
from models.model_train import deit_small_patch16_224 as vit_LRP_small_train



def model_env(pretrained=False,args  = None , hooks = False,  **kwargs):
    

    if hooks:
        if "size" in args.model_components:
            if args.model_components['size'] == 'base':
                return vit_LRP_base(
                    isWithBias           = args.model_components["isWithBias"],
                    isConvWithBias       = args.model_components["isConvWithBias"],

                    layer_norm           = args.model_components["norm"],
                    last_norm            = args.model_components["last_norm"],
                    attn_drop_rate       = args.model_components["attn_drop_rate"],
                    FFN_drop_rate        = args.model_components["FFN_drop_rate"],
                    patch_embed          = args.model_components["patch_embed"],
                    projection_drop_rate = args.model_components['projection_drop_rate'],


                    activation      = args.model_components["activation"],
                    attn_activation = args.model_components["attn_activation"],
                    num_classes     = args.nb_classes,
            )
            elif args.model_components['size'] == 'small':
                return vit_LRP_small(
                    isWithBias           = args.model_components["isWithBias"],
                    isConvWithBias       = args.model_components["isConvWithBias"],

                    layer_norm           = args.model_components["norm"],
                    last_norm            = args.model_components["last_norm"],
                    attn_drop_rate       = args.model_components["attn_drop_rate"],
                    FFN_drop_rate        = args.model_components["FFN_drop_rate"],
                    projection_drop_rate = args.model_components['projection_drop_rate'],

                    patch_embed          = args.model_components["patch_embed"],

                    activation      = args.model_components["activation"],
                    attn_activation = args.model_components["attn_activation"],
                    num_classes     = args.nb_classes,
            )
    
        return vit_LRP(
            isWithBias           = args.model_components["isWithBias"],
            isConvWithBias       = args.model_components["isConvWithBias"],

            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],

            patch_embed          = args.model_components["patch_embed"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )
    else:
        if "size" in args.model_components:
            if args.model_components['size'] == 'base':
                return vit_LRP_base_train(
                    isWithBias           = args.model_components["isWithBias"],
                    isConvWithBias       = args.model_components["isConvWithBias"],

                    layer_norm           = args.model_components["norm"],
                    last_norm            = args.model_components["last_norm"],
                    attn_drop_rate       = args.model_components["attn_drop_rate"],
                    FFN_drop_rate        = args.model_components["FFN_drop_rate"],
                    projection_drop_rate = args.model_components['projection_drop_rate'],

                    patch_embed          = args.model_components["patch_embed"],

                    activation      = args.model_components["activation"],
                    attn_activation = args.model_components["attn_activation"],
                    num_classes     = args.nb_classes,
            )
            elif args.model_components['size'] == 'small':
                return vit_LRP_small_train(
                    isWithBias           = args.model_components["isWithBias"],
                    isConvWithBias       = args.model_components["isConvWithBias"],

                    layer_norm           = args.model_components["norm"],
                    last_norm            = args.model_components["last_norm"],
                    attn_drop_rate       = args.model_components["attn_drop_rate"],
                    FFN_drop_rate        = args.model_components["FFN_drop_rate"],
                    projection_drop_rate = args.model_components['projection_drop_rate'],

                    patch_embed          = args.model_components["patch_embed"],

                    activation      = args.model_components["activation"],
                    attn_activation = args.model_components["attn_activation"],
                    num_classes     = args.nb_classes,
            )


        return vit_LRP_train(
            isWithBias           = args.model_components["isWithBias"],
            isConvWithBias       = args.model_components["isConvWithBias"],
            patch_embed          = args.model_components["patch_embed"],

            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],


            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )


   