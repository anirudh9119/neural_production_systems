from torch import nn
import torch
import math

from torch.distributions.multivariate_normal import MultivariateNormal

class MLP(nn.Module):
 def __init__(self, in_dim, out_dim):
     super().__init__()
     self.mlp = nn.Sequential(nn.Linear(in_dim, 64),
                              nn.ReLU(),
                              nn.Linear(64, out_dim))
 def forward(self, x):
     return self.mlp(x)

class SlotAttentionV2(nn.Module):
 def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128, out_dim = 64, in_dim = 64):
     super().__init__()
     self.num_slots = num_slots
     self.iters = iters
     self.eps = eps
     self.scale = dim ** -0.5
     self.dim = dim

     #self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
     #self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
     #self.slots_rep = nn.Linear(dim, 2*dim)
     #self.slots_rep_pr = nn.Linear(dim, 2*dim)
     print("using  iterations", self.iters)


     #self.slots_rep =  nn.Linear(dim, 2*dim)

     self.slots_rep = nn.Sequential(
       nn.Linear(dim, hidden_dim),
       nn.ReLU(inplace = True),
       nn.Linear(hidden_dim, 2*dim)
     )
     self.attn_images = []
     self.cur_image = None
     self.colors = [0,0,255]
     #self.slots_rep_pr = nn.Linear(dim, 2*dim)
     self.slots_rep_pr = nn.Sequential(
       nn.Linear(dim, hidden_dim),
       nn.ReLU(inplace = True),
       nn.Linear(hidden_dim, 2*dim)
     )

     self.to_q = nn.Linear(dim, dim)
     self.to_k = nn.Linear(in_dim, dim)
     self.to_v = nn.Linear(in_dim, dim)
     self.gru = nn.GRUCell(dim, dim)
     hidden_dim = max(dim, hidden_dim)
     self.mlp = nn.Sequential(
         nn.Linear(dim, hidden_dim),
         nn.ReLU(inplace = True),
         nn.Linear(hidden_dim, dim)
     )
     self.mlp_cast = MLP(dim, out_dim)
     self.norm_input  = nn.LayerNorm(in_dim)
     self.norm_slots  = nn.LayerNorm(dim)
     self.norm_pre_ff = nn.LayerNorm(dim)


 def reparameterize(self, mu, logvar, eps = None, deterministic=False):
     if not deterministic:
         std = logvar.mul(0.5).exp_()
         if eps is None:
             eps = std.data.new(std.size()).normal_()
         return eps.mul(std).add_(mu), eps
     else:
         return mu, eps

 def get_kl_loss(self, prior, post):
    prior = prior.reshape(-1, 2 * self.dim)
    post = post.reshape(-1, 2 * self.dim)
    mean_pr, std_pr = prior[:, : self.dim], prior[:, self.dim : 2 * self.dim]
    mean_po, std_po = post[:, : self.dim], post[:, self.dim : 2 * self.dim]
    log_var_po = std_po


    std_pr = std_pr.mul(0.5).exp_()
    std_po = std_po.mul(0.5).exp_()
    q1 = MultivariateNormal(loc=mean_pr, scale_tril=torch.diag_embed(std_pr))
    q2 = MultivariateNormal(loc=mean_po, scale_tril=torch.diag_embed(std_po))
    kl = torch.distributions.kl.kl_divergence(q2, q1)


    #mu, log_var = mean_po, log_var_po
    #kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    return kl

 def add_image(self):
         self.attn_images.append(self.cur_image)

 def reset_images(self):
         self.attn_images = []


 def forward(self, inputs, slots, prior_slots, num_slots = None):
         b, n, d_ = inputs.shape
         n_s = num_slots if num_slots is not None else self.num_slots
         d = slots.size(-1)
         inputs = self.norm_input(inputs)
         k, v = self.to_k(inputs), self.to_v(inputs)
         slots_mu_sigma = self.slots_rep(slots.reshape((slots.shape[0] * n_s, self.dim)))
         prior_mu_sigma = self.slots_rep_pr(prior_slots.reshape((slots.shape[0] * n_s, self.dim)).detach())
         prior = prior_mu_sigma.reshape(-1, 2 * self.dim)
         slots_mu_sigma = slots_mu_sigma.reshape((b, n_s, 2*self.dim))
         mu = slots_mu_sigma[:, :, :self.dim]
         sigma = slots_mu_sigma[:, :, self.dim: 2*self.dim]
         slots, eps = self.reparameterize(mu, sigma)
         kl_loss = 0
         for t in range(self.iters):
             slots_prev = slots
             #slots = self.norm_slots(slots)
             q = self.to_q(slots)
             dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
             attn = dots.softmax(dim=1) + self.eps
             self.attn = attn
             attn_value = attn.clone().detach()
             attn_image_ = torch.zeros(self.num_slots, 3, int(math.sqrt(attn_value.size(2))), int(math.sqrt(attn_value.size(2)))).to(attn_value.device)
             attn_image_ = attn_image_.view((self.num_slots, 3, -1))
             attn_image_[:,2,:] = 255
             for num_ in range(self.num_slots):
                attn_image_[num_, 2] = attn_image_[num_, 2] * attn_value[0, num_]
             attn_image_ = attn_image_.view(self.num_slots, 3, int(math.sqrt(attn_value.size(2))), int(math.sqrt(attn_value.size(2))))
             self.cur_image = attn_image_

             attn = attn / attn.sum(dim=-1, keepdim=True)
             updates = torch.einsum('bjd,bij->bid', v, attn)
             slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
             )
             slots = slots.reshape(b, -1, d)
             slots = slots + self.mlp(self.norm_pre_ff(slots))
             slots_mu_sigma = self.slots_rep(slots.reshape((slots.shape[0] * n_s, self.dim)))
             slots_mu_sigma = slots_mu_sigma.reshape((b, n_s, 2*self.dim))
             kl_loss += self.get_kl_loss(prior, slots_mu_sigma)
         return slots, kl_loss
