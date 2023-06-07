import wandb

def initialize_wandb(args, exp_name):
    wandb.login()

    wandb.init(
        project="flow_DiT",
        name=exp_name
    )
    wandb.config.update(args)

def log_loss_dict(loss_dict, steps):
    wandb.log({k: v for k, v in loss_dict.items()}, step=steps)

def log_images(samples, prefix, steps):
    wandb.log({f"{prefix}-samples": wandb.Image(samples)}, step=steps)