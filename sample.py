import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(precision=10)

class GenGaussLogLikelihood(nn.Module):
    def __init__(self, reduction='mean', alpha_eps=1e-4, beta_eps=1e-4, resi_min=1e-4, resi_max=1e3) -> None:
        super(GenGaussLogLikelihood, self).__init__()
        self.reduction = reduction
        self.alpha_eps = alpha_eps
        self.beta_eps = beta_eps
        self.resi_min = resi_min
        self.resi_max = resi_max

    def forward(self, mean: torch.Tensor, one_over_alpha: torch.Tensor, beta: torch.Tensor, target: torch.Tensor):
        one_over_alpha1 = one_over_alpha + self.alpha_eps
        beta1 = beta + self.beta_eps

        resi = torch.abs(mean - target)
        resi = torch.pow(resi * one_over_alpha1, beta1).clamp(min=self.resi_min, max=self.resi_max)

        log_one_over_alpha = torch.log(one_over_alpha1)
        log_beta = torch.log(beta1)
        lgamma_beta = torch.lgamma(torch.pow(beta1, -1))
        log_two = math.log(2)


        nll = resi - log_one_over_alpha + lgamma_beta - log_beta + log_two

        loglikelihood = -nll

        if self.reduction == 'mean':
            return loglikelihood.mean()
        elif self.reduction == 'sum':
            return loglikelihood.sum()
        elif self.reduction == 'none':
            return loglikelihood
        elif self.reduction == 'batchsum':
            return loglikelihood.sum(dim=1)
        else:
            print('Reduction not supported')
            return None

def generalized_gaussian_likelihood(loss):
    """損失から尤度に変換"""
    return torch.exp(-loss)

def plot_generalized_gaussian(mean, alpha, beta):
    """一般化ガウス分布をプロット"""
    x = torch.linspace(-3 + mean, 3 + mean, 1000)
    y = generalized_gaussian_pdf(x, mean, alpha, beta)

    plt.figure(figsize=(8, 6))
    plt.plot(x.numpy(), y.numpy(), label=f'mean={mean}, alpha={alpha}, beta={beta}')
    plt.title('Generalized Gaussian Distribution')
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.savefig('generalized_gaussian.png')  # グラフの保存
    print("Plot saved as 'generalized_gaussian.png'")

def generalized_gaussian_pdf(x, mean, alpha, beta):
    """一般化ガウス分布のPDF"""
    coeff = beta / (2 * alpha * math.gamma(1 / beta))
    exponent = -torch.pow(torch.abs(x - mean) / alpha, beta)
    return coeff * torch.exp(exponent)

def test_gen_gauss_loss():
    mean = torch.tensor([0.3])
    one_over_alpha = torch.tensor([30])
    beta = torch.tensor([0.5])
    target = torch.tensor([0])

    # loss_fn = GenGaussLossModified(reduction='mean', alpha_eps=0, beta_eps=0, resi_min=1e-4, resi_max=1e3)
    ll_fn = GenGaussLogLikelihood(reduction='mean', alpha_eps=0, beta_eps=0, resi_min=1e-4, resi_max=1e3)
    model_ll = ll_fn(mean, one_over_alpha, beta, target)

    # manual_loss = manual_generalized_gaussian_loss_torch(
    #     mean, one_over_alpha, beta, target
    # )
    manual_ll = generalized_gaussian_pdf(target, mean, 1.0 / one_over_alpha, beta)


    model_likelihood = model_ll.exp()
    print(f"Model Likelihood: {model_likelihood.item()}")

    manual_likelihood = manual_ll
    print(f"Manual Likelihood: {manual_likelihood.item()}")

    alpha = 1.0 / (one_over_alpha + 1e-4)
    plot_generalized_gaussian(mean.item(), alpha.item(), beta.item())

def manual_generalized_gaussian_loss_torch(mean, one_over_alpha, beta, target):
    alpha = 1.0 / one_over_alpha
    beta = beta
    resi = torch.abs(mean - target)

    log_likelihood = (
        torch.pow(resi / alpha, beta)
        - torch.log(beta) + math.log(2) + torch.log(alpha)
        + torch.lgamma(1 / beta)
    )

    # print(resi, log_one_over_alpha, lgamma_beta, log_beta, log_two)
    print(torch.pow(resi / alpha, beta), torch.log(1 / alpha), torch.lgamma(1 / beta), torch.log(beta), math.log(2))

    return log_likelihood.item()

# テスト実行
test_gen_gauss_loss()
