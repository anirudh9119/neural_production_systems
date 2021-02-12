from torch import nn
import torch

class ArgMax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, 1)
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        op = torch.zeros(input.size()).to(input.device)
        op.scatter_(1, idx[:, None], 1)
        ctx.save_for_backward(op)
        return op

    @staticmethod
    def backward(ctx, grad_output):
        op, = ctx.saved_tensors
        grad_input = grad_output * op
        return grad_input

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128, num_pixels = 64*64, extra_objective = False, lambda_=0.0, initial_lr=10):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.lambda_ = lambda_
        self.initial_lr = initial_lr
        print("Using lambda_, extra_objective, num_slots", extra_objective, self.lambda_, num_slots, self.iters, self.initial_lr)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.extra_objective = extra_objective

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)


    def bhattarcharya_distance(self, mean, var):

        distance = 0
        for i in range(self.num_slots):
            for j in range(i+1, self.num_slots):
               mean_diff = (mean[:, i, :] - mean[:, j, :])**2
               var_sum = (var[:, i, :] + var[:, j, :])
               var_mul = (var[:, i, :]  * var[:, j, :])
               first_term = mean_diff/(4*var_sum)
               second_term = torch.log((var_sum/2*torch.sqrt(var_mul)))
               distance += torch.exp(-(first_term+second_term))

        return distance


    def forward(self, inputs, positional_embeddings = None, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        num = torch.arange(n_s)
        combinations = torch.combinations(num, r = 2)

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        positional_embeddings = positional_embeddings.unsqueeze(1).repeat(1, n_s, 1, 1)

        alpha = torch.zeros(slots.size(0), n_s, inputs.size(1)).to(slots.device)
        if self.extra_objective:
            alpha.requires_grad = True

        for iter_ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale + alpha
            attn = (dots).softmax(dim=1) + self.eps

            if self.extra_objective:
                #dots_max = ArgMax().apply(attn).unsqueeze(-1)
                pos = positional_embeddings * attn.unsqueeze(3)
                num_pixel_per_slot = torch.sum(attn, dim = 2) #+ self.eps
                pos_sum_per_slot = torch.sum(pos, dim = 2)
                pos_mean_per_slot = pos_sum_per_slot / num_pixel_per_slot.unsqueeze(2)

                pos_sum_per_slot_ = pos_mean_per_slot.unsqueeze(2).repeat(1, 1, pos.size(2), 1)
                pos_var_sum_per_slot = (pos - pos_sum_per_slot_) ** 2
                pos_var_sum_per_slot = pos_var_sum_per_slot * attn.unsqueeze(3) # need to mask again since will lead to (0 - mu) ** 2
                pos_var_per_slot = torch.sum(pos_var_sum_per_slot, dim = 2) / num_pixel_per_slot.unsqueeze(2)

                first_term = self.bhattarcharya_distance(pos_mean_per_slot, pos_var_per_slot)
                forbenious_norm = torch.norm(alpha)

                objective = (first_term + self.lambda_ * forbenious_norm).mean()

                '''
                ########First Term ##########
                pos_mean_per_slot_1 = pos_mean_per_slot[:, combinations[:,0], :]
                pos_mean_per_slot_2 = pos_mean_per_slot[:, combinations[:,1], :]
                pos_var_per_slot_1 = pos_var_per_slot[:, combinations[:,0], :]
                pos_var_per_slot_2 = pos_var_per_slot[:, combinations[:,1], :]
                pos_mean_diff = (pos_mean_per_slot_1 - pos_mean_per_slot_2) ** 2
                pos_var_sum = 4*(pos_var_per_slot_1 + pos_var_per_slot_2)
                pos_mean_diff = torch.sum(pos_mean_diff/pos_var_sum, dim = 1)

                ########Second Term############
                pos_var_squared = pos_var_per_slot_1 + pos_var_per_slot_2
                pos_var_mul = 2*torch.sqrt(pos_var_per_slot_1 * pos_var_per_slot_2)
                pos_var = torch.log(pos_var_squared/pos_var_mul)
                pos_var = torch.sum(pos_var, dim = 1)
                forbenious_norm = torch.norm(alpha)

                objective = (torch.exp(-(pos_var + pos_mean_diff)) + self.lambda_ * forbenious_norm).mean()
                '''
                grad = torch.autograd.grad(objective, alpha, retain_graph = True)
                alpha = alpha - ((self.initial_lr*1.0)/(iter_+1))*grad[0]
                #alpha = alpha - grad[0]

            '''
            if False:  #self.extra_objective:
                dots_max = ArgMax().apply(attn).unsqueeze(-1)
                pos = positional_embeddings * dots_max #attn.unsqueeze(3)
                num_pixel_per_slot = torch.sum(dots_max, dim = 2) + self.eps
                pos_sum_per_slot = torch.sum(pos, dim = 2)
                #import ipdb
                #ipdb.set_trace()
                pos_mean_per_slot = pos_sum_per_slot / num_pixel_per_slot#.unsqueeze(2)
                pos_sum_per_slot_ = pos_mean_per_slot.unsqueeze(2).repeat(1, 1, pos.size(2), 1)
                pos_var_sum_per_slot = (pos - pos_sum_per_slot_) ** 2
                pos_var_sum_per_slot = pos_var_sum_per_slot * dots_max #attn.unsqueeze(3) # need to mask again since will lead to (0 - mu) ** 2
                pos_var_per_slot = torch.sum(pos_var_sum_per_slot, dim = 2) / num_pixel_per_slot#.unsqueeze(2)
                ########First Term ##########
                pos_mean_per_slot_1 = pos_mean_per_slot[:, combinations[:,0], :]
                pos_mean_per_slot_2 = pos_mean_per_slot[:, combinations[:,1], :]
                pos_var_per_slot_1 = pos_var_per_slot[:, combinations[:,0], :]
                pos_var_per_slot_2 = pos_var_per_slot[:, combinations[:,1], :]
                pos_mean_diff = 0.5 * (pos_mean_per_slot_1 - pos_mean_per_slot_2) ** 2
                pos_var_sum = pos_var_per_slot_1 + pos_var_per_slot_2
                pos_mean_diff = -torch.sum(pos_mean_diff/pos_var_sum, dim = 1)

                ########Second Term############

                pos_var_squared = (pos_var_per_slot_1 + pos_var_per_slot_2)**2
                pos_var_mul = pos_var_per_slot_1 * pos_var_per_slot_2
                pos_var = 2*torch.log(pos_var_squared/pos_var_mul)
                pos_var = torch.sum(pos_var, dim = 1)
                forbenious_norm = torch.norm(alpha)

                objective = (pos_var + pos_mean_diff + self.lambda_ * forbenious_norm).mean()
                grad = torch.autograd.grad(objective, alpha, retain_graph = True)
                alpha = alpha - grad[0]
            '''

            #print(alpha[0,:,96])

            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


if __name__ == '__main__':
    sa = SlotAttention(5, 64, num_pixels = 128, extra_objective = True).cuda()

    x = torch.rand(2, 128, 64).cuda()

    positional_embeddings = torch.rand(2, 128, 64).cuda()
    out = sa(x, positional_embeddings = positional_embeddings)

