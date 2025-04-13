import torch
from torch import nn
from models_rad import RESNET3d


def generate_model(args):


    if args.model == 'RESNET3d':
        assert args.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if args.model_depth == 10:
            model = RESNET3d.resnet10(
                sample_input_W=args.input_W,
                sample_input_H=args.input_H,
                sample_input_D=args.input_D,
                shortcut_type=args.resnet_shortcut,
                no_cuda=args.no_cuda,
                num_seg_classes=args.n_seg_classes)
        elif args.model_depth == 18:
            model = RESNET3d.resnet18(
                sample_input_W=args.input_W,
                sample_input_H=args.input_H,
                sample_input_D=args.input_D,
                shortcut_type=args.resnet_shortcut,
                no_cuda=args.no_cuda,
                num_seg_classes=args.n_seg_classes)
        elif args.model_depth == 34:
            model = RESNET3d.resnet34(
                sample_input_W=args.input_W,
                sample_input_H=args.input_H,
                sample_input_D=args.input_D,
                shortcut_type=args.resnet_shortcut,
                no_cuda=args.no_cuda,
                num_seg_classes=args.n_seg_classes)
        elif args.model_depth == 50:
            model = RESNET3d.resnet50(
                sample_input_W=args.input_W,
                sample_input_H=args.input_H,
                sample_input_D=args.input_D,
                shortcut_type=args.resnet_shortcut,
                no_cuda=args.no_cuda,
                num_seg_classes=args.n_seg_classes
            )
        elif args.model_depth == 101:
            model = RESNET3d.resnet101(
                sample_input_W=args.input_W,
                sample_input_H=args.input_H,
                sample_input_D=args.input_D,
                shortcut_type=args.resnet_shortcut,
                no_cuda=args.no_cuda,
                num_seg_classes=args.n_seg_classes)
        elif args.model_depth == 152:
            model = RESNET3d.resnet152(
                sample_input_W=args.input_W,
                sample_input_H=args.input_H,
                sample_input_D=args.input_D,
                shortcut_type=args.resnet_shortcut,
                no_cuda=args.no_cuda,
                num_seg_classes=args.n_seg_classes)
        elif args.model_depth == 200:
            model = RESNET3d.resnet200(
                sample_input_W=args.input_W,
                sample_input_H=args.input_H,
                sample_input_D=args.input_D,
                shortcut_type=args.resnet_shortcut,
                no_cuda=args.no_cuda,
                num_seg_classes=args.n_seg_classes)
    
    if not args.no_cuda:
        if len(args.gpu_id) > 1:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=args.gpu_id)
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id[0])
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()
    
    # load pretrain
    if args.pretrain_path:
        print ('loading pretrained model {}'.format(args.pretrain_path))
        pretrain = torch.load(args.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
        if model.state_dict():
            print("Pretrained weights loaded successfully.")
        else:
            print("Failed to load pretrained weights.")
        # new_parameters = [] 
        # for pname, p in model.named_parameters():
        #     for layer_name in args.new_layer_names:
        #         if pname.find(layer_name) >= 0:
        #             new_parameters.append(p)
        #             break

        # new_parameters_id = list(map(id, new_parameters))
        # base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        # parameters = {'base_parameters': base_parameters, 
        #               'new_parameters': new_parameters}
        print(model)
        return model

    return model, model.parameters()