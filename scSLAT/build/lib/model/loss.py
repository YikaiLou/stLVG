r"""
Loss function
"""
import torch

#损失函数MSE，衡量重构模型对嵌入特征的重构效果
def feature_reconstruct_loss(embd:torch.Tensor,
                             x:torch.Tensor,
                             recon_model:torch.nn.Module
    ) -> torch.Tensor:
    r"""
    Reconstruction loss (MSE)
    
    Parameters
    ----------
    embd
        embd of a cell
    x
        input 
    recon_model
        reconstruction model
    """
    recon_x = recon_model(embd)
    return torch.norm(recon_x - x, dim=1, p=2).mean()