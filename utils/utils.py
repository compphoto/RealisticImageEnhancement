from datetime import datetime

def normalize(x):
    return (x - 0.5) / 0.5

def create_exp_name(args, prefix='EditNet'):

    components = []
    file_name = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    components.append(f"{prefix}")
    components.append(f"exp_{file_name}")
    components.append(f"{args.expdscp}")
    components.append(f"sldir{args.sal_loss_type}")
    components.append(f"lrprmtr_{args.lr_parameters}")
    components.append(f"btar{args.beta_r}")
    components.append(f"wsal{args.w_sal}")

    


    name = "_".join(components)
    return name

def create_exp_name_disc(args, prefix='RealismNet'):

    components = []
    file_name = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    components.append(f"{prefix}")
    components.append(f"exp_{file_name}")
    components.append(f"{args.expdscp}")
    components.append(f"lrd_{args.lr_d}")

    name = "_".join(components)
    return name
