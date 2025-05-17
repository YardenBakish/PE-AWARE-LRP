import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from baselines.ViT.layer_helpers import to_2tuple

from torch.nn.utils.parametrizations import weight_norm
from zennit.rules import  Gamma

import math

__all__ = ['forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'GELU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide', 'einsum', 'Softmax', 'IndexSelect',
           'LayerNorm', 'AddEye','BatchNorm1D' ,'RMSNorm' , 'Softplus', 'UncenteredLayerNorm', 'Sigmoid', 'SigmoidAttention', 'ReluAttention',
           'Sparsemax',
           'RepBN',
           'SiLU', 'WeightNormLinear', 'NormalizedLayerNorm' , 'NormalizedConv2d', 
           'CustomLRPLayerNorm', 'CustomLRPRMSNorm', 'NormalizedReluAttention', 'CustomLRPBatchNorm',
           'SNLinear', 'SNConv2d']


def _stabilize(input, epsilon=1e-6, inplace=False):
    """
    Stabilize the input by adding a small value to it
    """
    if epsilon == None:
        epsilon = 1e-6
    if inplace:
        return input.add_(epsilon)
    else:
        return input + epsilon

def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        return R
    
    def m_relprop(self, R,pred,  alpha):
        return R
    def RAP_relprop(self, R_p):
        return R_p

class RelPropSimple(RelProp):
    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs
    
    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = self.forward(self.X)
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, self.X, Sp)
            if torch.is_tensor(self.X) == False:
                Rp = []
                Rp.append(self.X[0] * Cp[0])
                Rp.append(self.X[1] * Cp[1])
            else:
                Rp = self.X * (Cp[0])
            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp

class AddEye(RelPropSimple):
    # input of shape B, C, seq_len, seq_len
    def forward(self, input):
        return input + torch.eye(input.shape[2]).expand_as(input).to(input.device)

class ReLU(nn.ReLU, RelProp):
    pass

class SiLU(nn.SiLU, RelProp):
    pass

class GELU(nn.GELU, RelProp):
    pass


class Softplus(nn.Softplus, RelProp):
    pass

class Softmax(nn.Softmax, RelProp):
    pass

class LayerNorm(nn.LayerNorm, RelProp):
    pass


class RMSNorm(nn.RMSNorm, RelProp):
    pass



class Sigmoid(nn.Sigmoid, RelProp):
    pass

class SigmoidAttention(RelProp):
    def __init__(self, n= 197):
        super(SigmoidAttention, self).__init__()
        self.b = -math.log(n)
        self.act_variant = Sigmoid()
    
    def forward(self,x):
        return  self.act_variant(x + self.b)


class ReluAttention(nn.Module):
    def __init__(self, n= 197):
        super(ReluAttention, self).__init__()
        self.seqlen = n ** -1
        self.act_variant = ReLU()
    
    def forward(self,x):
    
        return self.act_variant(x) * self.seqlen
    
    def relprop(self, cam, **kwargs):
        return self.act_variant.relprop(cam, **kwargs)

    def RAP_relprop(self,R):
        return R
    
class BatchNorm1D(nn.BatchNorm1d, RelProp):
    pass

class Dropout(nn.Dropout, RelProp):
    pass


class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass

class LayerNorm(nn.LayerNorm, RelProp):
    pass

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelPropSimple):
    pass


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):

        if epsilon_rule and default_op == False:
            relevance_norm = R / _stabilize(self.X[0] + self.X[1], epsilon=1e-6, inplace=False)
            relevance_a = relevance_norm * self.X[0]
            relevance_b = relevance_norm * self.X[1]
            return [relevance_a, relevance_b]


        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum(dim=(-2,-1))  # Shape: [batch_size]
        b_sum = b.sum(dim=(-2,-1))  # Shape: [batch_size]


        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()).unsqueeze(-1).unsqueeze(-1) * R.sum(dim=(-2,-1), keepdim=True)
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()).unsqueeze(-1).unsqueeze(-1) * R.sum(dim=(-2,-1), keepdim=True)

        a = a * safe_divide(a_fact, a.sum(dim=(-2,-1), keepdim=True))
        b = b * safe_divide(b_fact, b.sum(dim=(-2,-1), keepdim=True))

        outputs = [a, b]

        return outputs
    
    def RAP_relprop(self, R):


        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum(dim=(-2,-1))  # Shape: [batch_size]
        b_sum = b.sum(dim=(-2,-1))  # Shape: [batch_size]


        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()).unsqueeze(-1).unsqueeze(-1) * R.sum(dim=(-2,-1), keepdim=True)
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()).unsqueeze(-1).unsqueeze(-1) * R.sum(dim=(-2,-1), keepdim=True)

        a = a * safe_divide(a_fact, a.sum(dim=(-2,-1), keepdim=True))
        b = b * safe_divide(b_fact, b.sum(dim=(-2,-1), keepdim=True))

        outputs = [a, b]

        return outputs

class einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation
    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)
    
    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
      if epsilon_rule and default_op == False:
        relevance_norm = R / _stabilize(self.Y * 2, 1e-6, inplace=False)
        flag_transposed = False
        if self.X[1].shape[2] != self.X[0].shape[3]:
          self.X[1] = self.X[1].transpose(-1, -2)
          flag_transposed = True

        relevance_a = torch.matmul(relevance_norm, self.X[1].transpose(-1, -2)).mul_(self.X[0])
        relevance_b = torch.matmul(self.X[0].transpose(-1, -2), relevance_norm).mul_(self.X[1])
        
        if flag_transposed:
          relevance_b = relevance_b.transpose(-1, -2)

        return [2*relevance_a, 2*relevance_b]
      else:
        return super().relprop(R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule)

class IndexSelect(RelProp):
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha, epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs


    def RAP_relprop(self, R):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs
    
class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R
    def RAP_relprop(self, R_p):

        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R_p, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C
        return R
        def backward(R_p):
            Z = []
            for _ in range(self.num):
                Z.append(self.X)

            Spp = []
            Spn = []

            for z, rp in zip(Z, R_p):
                Spp.append(safe_divide(torch.clamp(rp, min=0), z))
                Spn.append(safe_divide(torch.clamp(rp, max=0), z))
            print(len(Spp))
            print(self.X.shape)
            print(len(Z))
            print("TTATA\n\n")


            Cpp = self.gradprop(Z, self.X, Spp)[0]

            Cpn = self.gradprop(Z, self.X, Spn)[0]
            print("TTATA\n\n")

            Rp = self.X * (Cpp * Cpn)

            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            print(len(tmp_R_p[0]))
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp
    

class Cat(RelProp):
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs

    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = self.forward(self.X, self.dim)
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, self.X, Sp)

            Rp = []

            for x, cp in zip(self.X, Cp):
                Rp.append(x * (cp))


            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp

class CustomLRPLayerNorm(nn.LayerNorm, RelProp):
    def forward(self, x):
        with torch.enable_grad():

            mean = x.mean(dim=-1, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            std = (var + self.eps).sqrt()
            y = (x - mean) / std.detach() # detach std operation will remove it from computational graph i.e. identity rule on x/std
            if self.weight is not None:
                y *= self.weight
            if self.bias is not None:
                y += self.bias

            self.output11 = y
       
        return y

    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        Z = self.forward(self.X)
        relevance_norm = R / _stabilize(Z, self.eps, False)
        grads= torch.autograd.grad(self.output11, self.X, relevance_norm)[0]
        return grads*self.X


    def RAP_relprop(self, Rp):
        Z = self.forward(self.X)
        relevance_norm = Rp / _stabilize(Z, self.eps, False)
        grads= torch.autograd.grad(self.output11, self.X, relevance_norm)[0]
        return grads*self.X

'''
class CustomLRPBatchNorm(nn.BatchNorm1d, RelProp):
    def forward(self, x):
        with torch.enable_grad():
            # Compute current batch statistics
            mean = x.mean(dim=[0, 2])  # Mean across batch and sequence length
            var = ((x - mean.view(1, -1, 1)) ** 2).mean(dim=[0, 2])
            
            # Update running averages
            if self.track_running_stats:
                with torch.no_grad():
                    # Exponential moving average update
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            # Use running stats or current batch stats
            use_mean = self.running_mean.view(1, -1, 1) if self.track_running_stats else mean.view(1, -1, 1)
            use_var = self.running_var.view(1, -1, 1) if self.track_running_stats else var.view(1, -1, 1)
            
            # Compute std and detach it
            std = (use_var + self.eps).sqrt().detach()
            
            # Normalize with detached std
            y = (x - use_mean) / std
            
            # Apply learnable scale and shift
            if self.weight is not None:
                y *= self.weight.view(1, -1, 1)
            if self.bias is not None:
                y += self.bias.view(1, -1, 1)
            
            # Store for relevance propagation
            self.X = x
            self.output11 = y
        
        return y

    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        Z = self.forward(self.X)
        relevance_norm = R[0] / _stabilize(Z, self.eps, False)
        grads= torch.autograd.grad(self.output11, self.X, relevance_norm)[0]
        return grads*self.X

'''



class CustomLRPBatchNorm(nn.BatchNorm1d, RelProp):
    def forward(self, x):
      with torch.enable_grad():

        mean = self.running_mean
        var = self.running_var

        std = (var.view(1, -1, 1) + self.eps).sqrt()
        x_normalized = (x - mean.view(1, -1, 1)) / std

        # Scale and shift
        if self.weight is not None:
          x_normalized = x_normalized * self.weight.view(1, -1, 1)
        if self.bias is not None:
          x_normalized = x_normalized + self.bias.view(1, -1, 1)
          # Store for LRP
      self.X = x
      self.output11 = x_normalized

      return x_normalized
   
    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):

     
        Z = self.forward(self.X)
        var = self.running_var

        #std = (var.view(1, -1, 1) + self.eps).sqrt()
        #print(std.shape)

        #w = self.weight.unsqueeze(0).unsqueeze(-1)
        #print(w.shape)

        relevance_norm = R 
      
        grads= torch.autograd.grad(Z, self.X, relevance_norm)[0]
        return (grads*self.X) 









class CustomLRPRMSNorm(nn.RMSNorm, RelProp):
    def forward(self, x):
        with torch.enable_grad():
            
            var = (x  ** 2).mean(dim=-1, keepdim=True)
            if self.eps:
                var = var + self.eps
            std = (var).sqrt()
            y = (x ) / std.detach() # detach std operation will remove it from computational graph i.e. identity rule on x/std
            
            if self.weight is not None:
                y *= self.weight
            
            self.output11 = y
        return y

    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        Z = self.forward(self.X)
        relevance_norm = R / _stabilize(Z, self.eps, False)
        grads= torch.autograd.grad(self.output11, self.X, relevance_norm)[0]
        return grads*self.X






class Sequential(nn.Sequential):
    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha, epsilon_rule, gamma_rule, default_op, conv_gamma_rule)
        return R
    def RAP_relprop(self, Rp):
        for m in reversed(self._modules.values()):
            Rp = m.RAP_relprop(Rp)
        return Rp


class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def relprop(self, R, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R


class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha, epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        if gamma_rule:
          layer = nn.Linear(self.in_features, self.out_features)
          layer.to(self.weight.device)
          with torch.no_grad():
              layer.weight.copy_(self.weight)
              if self.bias is not None:
                layer.bias.copy_(self.bias)
          
          rule = Gamma(gamma_rule)  #DEFAULT: 0.05
          handles = rule.register(layer)
          output = layer(self.X)
          attribution, = torch.autograd.grad(output, self.X, grad_outputs=R)
          handles.remove()
          return attribution
        
        if epsilon_rule:
            relevance_norm = R / _stabilize(self.Y, 1e-8, inplace=False) 
            Z = F.linear(self.X, self.weight)
            grads = torch.autograd.grad(Z, self.X, relevance_norm)[0]
            return self.X * grads
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.linear(x1, w1)
                Z2 = F.linear(x2, w2)
                S1 = safe_divide(R, Z1 + Z2)
                S2 = safe_divide(R, Z1 + Z2)
                C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
                C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances

            return R

    def RAP_relprop(self, R_p):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=-1, keepdim=True)) * torch.ne(R, 0).type(R.type())
            K = R - shift
            return K
        def pos_prop(R, Za1, Za2, x1):
            R_pos = torch.clamp(R, min=0)
            R_neg = torch.clamp(R, max=0)
            S1 = safe_divide((R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide((R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1n = x1 * self.gradprop(Za1, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n

            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=-1,keepdim=True)-R.sum(dim=-1,keepdim=True))
            return C
        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            Za1 = F.linear(x1, w1) * R_nonzero
            Za2 = - F.linear(x1, w2) * R_nonzero

            Zb1 = - F.linear(x2, w1) * R_nonzero
            Zb2 = F.linear(x2, w2) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)

            return C1 + C2
        def first_prop(pd, px, nx, pw, nw):
            b = 0
            if self.bias is not None:
                b = self.bias
            Rpp = F.linear(px, pw) * pd
            Rpn = F.linear(px, nw) * pd
            Rnp = F.linear(nx, pw) * pd
            Rnn = F.linear(nx, nw) * pd
            Pos = (Rpp + Rnn).sum(dim=-1, keepdim=True)
            Neg = (Rpn + Rnp).sum(dim=-1, keepdim=True)

            Z1 = F.linear(px, pw)
            Z2 = F.linear(px, nw)
            Z3 = F.linear(nx, pw)
            Z4 = F.linear(nx, nw)

            S1 = safe_divide(Rpp, Z1)
            S2 = safe_divide(Rpn, Z2)
            S3 = safe_divide(Rnp, Z3)
            S4 = safe_divide(Rnn, Z4)
            C1 = px * self.gradprop(Z1, px, S1)[0]
            C2 = px * self.gradprop(Z2, px, S2)[0]
            C3 = nx * self.gradprop(Z3, nx, S3)[0]
            C4 = nx * self.gradprop(Z4, nx, S4)[0]
            bp = b * pd * safe_divide(Pos, Pos + Neg)
            bn = b * pd * safe_divide(Neg, Pos + Neg)
            Sb1 = safe_divide(bp, Z1)
            Sb2 = safe_divide(bn, Z2)
            Cb1 = px * self.gradprop(Z1, px, Sb1)[0]
            Cb2 = px * self.gradprop(Z2, px, Sb2)[0]
            return C1 + C4 + Cb1 + C2 + C3 + Cb2
        def backward(R_p, px, nx, pw, nw):
            # dealing bias
            # if torch.is_tensor(self.bias):
            #     bias_p = self.bias * R_p.ne(0).type(self.bias.type())
            #     R_p = R_p - bias_p

            Rp = f(R_p, pw, nw, px, nx)

            # if torch.is_tensor(self.bias):
            #     Bp = f(bias_p, pw, nw, px, nx)
            #
            #     Rp = Rp + Bp
            return Rp
        def redistribute(Rp_tmp):
            Rp = torch.clamp(Rp_tmp, min=0)
            Rn = torch.clamp(Rp_tmp, max=0)
            R_tot = (Rp - Rn).sum(dim=-1, keepdim=True)
            Rp_tmp3 = safe_divide(Rp, R_tot) * (Rp + Rn).sum(dim=-1, keepdim=True)
            Rn_tmp3 = -safe_divide(Rn, R_tot) * (Rp + Rn).sum(dim=-1, keepdim=True)
            return Rp_tmp3 + Rn_tmp3
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        X = self.X
        px = torch.clamp(X, min=0)
        nx = torch.clamp(X, max=0)
        if torch.is_tensor(R_p) == True and R_p.max() == 1:  ## first propagation
            pd = R_p

            Rp_tmp = first_prop(pd, px, nx, pw, nw)
            A =  redistribute(Rp_tmp)

            return A
        else:
            Rp = backward(R_p, px, nx, pw, nw)


        return Rp



'''
class NormalizedReluAttention(nn.Module):
    def __init__(self, n= 197):
        super(NormalizedReluAttention, self).__init__()
        self.seqlen = n ** -1
        self.act_variant = ReLU()
    
    def forward(self,x):
      x = self.act_variant(x)
      row_sums = x.sum(dim=3, keepdim=True)  
      normalized_attention_map = x / (row_sums + 1e-6)

      return normalized_attention_map # * self.seqlen
    
    def relprop(self, cam, **kwargs):
        return self.act_variant.relprop(cam, **kwargs)
    
'''



class NormalizedReluAttention(nn.Module):
    def __init__(self, n= 197):
        super(NormalizedReluAttention, self).__init__()
        self.seqlen = n ** -1
        self.act_variant = ReLU()
    
    def forward(self,x):
      with torch.enable_grad():
        self.X  = x
        x = self.act_variant(x)
        row_sums = x.sum(dim=3, keepdim=True)  
        normalized_attention_map = x / (row_sums + 1e-6).detach()
        self.Y = normalized_attention_map
      return normalized_attention_map # * self.seqlen
    
    def relprop(self, cam, **kwargs):
        Z = self.forward(self.X)
        relevance_norm = cam / _stabilize(Z, 1e-6, False)
        grads= torch.autograd.grad(self.Y, self.X, relevance_norm)[0]

        return grads*self.X
    





class WeightNormLinear(Linear):
  def __init__(self, in_features, out_features, bias=True):
    super().__init__(in_features, out_features, bias)
    weight_norm(self, name='weight')
  
  def relprop(self, R, alpha, epsilon_rule, gamma_rule, default_op, conv_gamma_rule):


    if gamma_rule:
        layer = nn.Linear(self.in_features, self.out_features)
        layer.to(self.weight.device)
        with torch.no_grad():
            layer.weight.copy_(self.weight)
            if self.bias is not None:
                layer.bias.copy_(self.bias)
          
        rule = Gamma(gamma_rule)
        handles = rule.register(layer)
        output = layer(self.X)
        attribution, = torch.autograd.grad(output, self.X, grad_outputs=R)
        handles.remove()

        return attribution

    if epsilon_rule:
        relevance_norm = R #/ _stabilize(self.Y, 1e-8, inplace=False) FIXEME: breaks everything
        weight_g = self.parametrizations.weight.original0  # Weight scale
        weight_v = self.parametrizations.weight.original1  # Weight directio
    
        weight =  weight_g * (weight_v / torch.norm(weight_v, dim=1)[:,None])
        Z =  F.linear(self.X, weight)
        grads = torch.autograd.grad(Z, self.X, relevance_norm)[0]
        return self.X * grads
    else:
        beta = alpha - 1

        weight_g = self.parametrizations.weight.original0  # Weight scale
        weight_v = self.parametrizations.weight.original1  # Weight directio
    
        weight =  weight_g * (weight_v / torch.norm(weight_v, dim=1)[:,None])
    
        # Clamp weights and input
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight , max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]
            return C1 + C2
        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R


class NormalizedLayerNorm(LayerNorm):
  def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
    super().__init__(normalized_shape, eps, elementwise_affine, bias)
    weight_norm(self, name='weight')


class Conv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, conv_prop_rule, alpha,epsilon_rule, gamma_rule, default_op, conv_gamma_rule):
        '''
        relevance_norm = R / _stabilize(self.Y, 50, inplace=False)
        Z = F.conv2d(self.X, self.weight, bias=None, stride=self.stride, padding=self.padding)
        #grads = torch.autograd.grad(Z, self.X, relevance_norm)[0]
        grads = self.gradprop2(relevance_norm, self.weight)[0]
        
        return self.X * grads
        '''
        if conv_prop_rule == "None":
            print("conv layer wasnt given a propagation rule")
            exit(1)
        if conv_prop_rule == "gammaConv":
         hasBias = self.bias is not None
         layer = nn.Conv2d(self.in_channels, self.out_channels, stride=self.stride,kernel_size = self.kernel_size, bias=hasBias, padding = self.padding)
         layer.to(self.weight.device)
         with torch.no_grad():
             layer.weight.copy_(self.weight)
             if self.bias is not None:
                 layer.bias.copy_(self.bias)
         rule = Gamma(conv_gamma_rule)
         handles = rule.register(layer)
         output = layer(self.X)
         attribution, = torch.autograd.grad(output, self.X, grad_outputs=R)
         handles.remove()

         return attribution
        
        if conv_prop_rule == "alphaConv":
            if self.X.shape[1] == 3:
                pw = torch.clamp(self.weight, min=0)
                nw = torch.clamp(self.weight, max=0)
                X = self.X
                L = self.X * 0 + \
                    torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                              keepdim=True)[0]
                H = self.X * 0 + \
                    torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                              keepdim=True)[0]
                Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                     torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                     torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

                S = R / Za
                C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
                R = C
            else:
                beta = alpha - 1
                pw = torch.clamp(self.weight, min=0)
                nw = torch.clamp(self.weight, max=0)
                px = torch.clamp(self.X, min=0)
                nx = torch.clamp(self.X, max=0)

                def f(w1, w2, x1, x2):
                    Z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
                    Z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
                    S1 = safe_divide(R, Z1)
                    S2 = safe_divide(R, Z2)
                    C1 = x1 * self.gradprop(Z1, x1, S1)[0]
                    C2 = x2 * self.gradprop(Z2, x2, S2)[0]
                    return C1 + C2

                activator_relevances = f(pw, nw, px, nx)
                inhibitor_relevances = f(nw, pw, px, nx)

                R = alpha * activator_relevances - beta * inhibitor_relevances
            return R
    
    def RAP_relprop(self, R_p):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=[1,2,3], keepdim=True)) * torch.ne(R, 0).type(R.type())
            K = R - shift
            return K
        def pos_prop(R, Za1, Za2, x1):
            R_pos = torch.clamp(R, min=0)
            R_neg = torch.clamp(R, max=0)
            S1 = safe_divide((R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide((R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za2)
            C1n = x1 * self.gradprop(Za2, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n
            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=[1,2,3], keepdim=True) - R.sum(dim=[1,2,3], keepdim=True))
            return C
        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            Za1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding) * R_nonzero
            Za2 = - F.conv2d(x1, w2, bias=None, stride=self.stride, padding=self.padding) * R_nonzero

            Zb1 = - F.conv2d(x2, w1, bias=None, stride=self.stride, padding=self.padding) * R_nonzero
            Zb2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)
            return C1 + C2
        def backward(R_p, px, nx, pw, nw):

            # if torch.is_tensor(self.bias):
            #     bias = self.bias.unsqueeze(-1).unsqueeze(-1)
            #     bias_p = safe_divide(bias * R_p.ne(0).type(self.bias.type()),
            #                          R_p.ne(0).type(self.bias.type()).sum(dim=[2, 3], keepdim=True))
            #     R_p = R_p - bias_p

            Rp = f(R_p, pw, nw, px, nx)

            # if torch.is_tensor(self.bias):
            #     Bp = f(bias_p, pw, nw, px, nx)
            #
            #     Rp = Rp + Bp
            return Rp
        def final_backward(R_p, pw, nw, X1):
            X = X1
            L = X * 0 + \
                torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = X * 0 + \
                torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding)

            Sp = safe_divide(R_p, Za)

            Rp = X * self.gradprop2(Sp, self.weight) - L * self.gradprop2(Sp, pw) - H * self.gradprop2(Sp, nw)
            return Rp
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        if self.X.shape[1] == 3:
            Rp = final_backward(R_p, pw, nw, self.X)
        else:
            Rp = backward(R_p, px, nx, pw, nw)
        return Rp


class NormalizedConv2d(Conv2d):
  def __init__(self, in_chans, embed_dim, kernel_size=None, stride=None):
    super().__init__(in_chans, embed_dim, kernel_size, stride)
    weight_norm(self, name='weight')


class RepBN(nn.Module):
    def __init__(self, normalized_shape, batchLayer = BatchNorm1D ):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = batchLayer(normalized_shape)
        self.add = Add()
        self.clone = Clone()

    def forward(self, x):
        x1, x2 = self.clone(x, 2)

        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)

        x = self.add([self.bn(x1), self.alpha * x2])  

        x = x.transpose(1, 2)

        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1,2)
        (cam1, cam2) = self.add.relprop(cam, **kwargs)
        cam2 /= self.alpha
        cam1  = self.bn.relprop(cam, **kwargs)

        cam1 = cam1.transpose(1,2)
        cam2 = cam2.transpose(1,2)

        cam = self.clone.relprop((cam1, cam2), **kwargs)

        return cam


class UncenteredLayerNorm(RelProp):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine' ]
    normalized_shape: tuple
    eps: float
    elementwise_affine: bool



    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, has_bias=True, center=True):
        super(UncenteredLayerNorm, self).__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.has_bias = has_bias
        self.has_center = center
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(normalized_shape))
            if has_bias:
                self.bias = nn.Parameter(torch.empty(normalized_shape))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.has_center:
            # Standard LayerNorm behavior with centering
            mean = x.mean(-1, keepdim=True)
            var = x.pow(2).mean(-1, keepdim=True) - mean.pow(2)
            x = (x - mean) / torch.sqrt(var + self.eps)
        else:
            # Uncentered version
            var = x.pow(2).mean(-1, keepdim=True)
            x = x / torch.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            if self.bias is not None:
                x = self.weight * x + self.bias
            else:
                x = self.weight * x
            
        return x

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}, ' \
            'bias={bias}'.format(**self.__dict__)
    






####################################################################
## SPARSEMAX
####################################################################


def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)


def _sparsemax_threshold_and_support(X, dim=-1, k=None):
    if k is None or k >= X.shape[dim]:  # do full sort
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size




#FIXME: acc results are good. consider implementing relprop based on Linear (positive and negative contributions)
class SparsemaxFunction(Function):
    @classmethod
    def forward(cls, ctx, X, dim=-1, k=None):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as softmax
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
        output = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output, supp_size

    @classmethod
    def backward(cls, ctx, grad_output, supp):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None, None



def sparsemax(X, dim=-1, k=None, return_support_size=False):
    P, support = SparsemaxFunction.apply(X, dim, k)
    if return_support_size:
        return P, support
    return P




class Sparsemax(RelProp):
    def __init__(self, dim=-1, k=None, return_support_size=False):
        self.dim = dim
        self.k = k
        self.return_support_size = return_support_size
        super(Sparsemax, self).__init__()

    def forward(self, X):
        return sparsemax(X, dim=self.dim, k=self.k, return_support_size=self.return_support_size)













####################################################################
## Sigma Reparam Layers
####################################################################

class SpectralNormedWeight(nn.Module):
    """SpectralNorm Layer. First sigma uses SVD, then power iteration."""

    def __init__(
        self,
        weight: torch.Tensor,
    ):
        super().__init__()
        self.weight = weight
        with torch.no_grad():
            _, s, vh = torch.linalg.svd(self.weight, full_matrices=False)

        self.register_buffer("u", vh[0])
        self.register_buffer("spectral_norm", s[0] * torch.ones(1))

    def get_sigma(self, u: torch.Tensor, weight: torch.Tensor):
        with torch.no_grad():
            v = weight.mv(u)
            v = nn.functional.normalize(v, dim=0)
            u = weight.T.mv(v)
            u = nn.functional.normalize(u, dim=0)
            if self.training:
                self.u.data.copy_(u)

        return torch.einsum("c,cd,d->", v, weight, u) 

    def forward(self):
        """Normalize by largest singular value and rescale by learnable."""
        sigma = self.get_sigma(u=self.u, weight=self.weight)
        if self.training:
            self.spectral_norm.data.copy_(sigma)

        return self.weight / sigma
    

class SNLinear(nn.Linear):
    """Spectral Norm linear from sigmaReparam.

    Optionally, if 'stats_only' is `True`,then we
    only compute the spectral norm for tracking
    purposes, but do not use it in the forward pass.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_multiplier: float = 1.0,
        stats_only: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.stats_only = stats_only
        self.init_multiplier = init_multiplier

        self.init_std = 0.02 * init_multiplier
        nn.init.trunc_normal_(self.weight, std=self.init_std)

        # Handle normalization and add a learnable scalar.
        self.spectral_normed_weight = SpectralNormedWeight(self.weight)
        sn_init = self.spectral_normed_weight.spectral_norm

        # Would have set sigma to None if `stats_only` but jit really disliked this
        self.sigma = (
            torch.ones_like(sn_init)
            if self.stats_only
            else nn.Parameter(
                torch.zeros_like(sn_init).copy_(sn_init), requires_grad=True
            )
        )

        self.register_buffer("effective_spectral_norm", sn_init)
        self.update_effective_spec_norm()

    def update_effective_spec_norm(self):
        """Update the buffer corresponding to the spectral norm for tracking."""
        with torch.no_grad():
            s_0 = (
                self.spectral_normed_weight.spectral_norm
                if self.stats_only
                else self.sigma
            )
            self.effective_spectral_norm.data.copy_(s_0)

    def get_weight(self):
        """Get the reparameterized or reparameterized weight matrix depending on mode
        and update the external spectral norm tracker."""
        normed_weight = self.spectral_normed_weight()
        self.update_effective_spec_norm()
        return self.weight if self.stats_only else normed_weight * self.sigma

    def forward(self, inputs: torch.Tensor):
        self.X  = inputs
        weight = self.get_weight()
        res =  F.linear(inputs, weight, self.bias)
        self.Y = res
        return res
    
    def relprop(self, R, alpha, epsilon_rule, gamma_rule, default_op, conv_gamma_rule):

        if epsilon_rule:
            relevance_norm = R / _stabilize(self.Y, 1e-8, inplace=False)
            Z = self.forward(self.X)
            grads = torch.autograd.grad(Z, self.X, relevance_norm)[0]
            return self.X * grads
        else:
            weight = self.get_weight()
            beta = alpha - 1
            pw = torch.clamp(weight, min=0)
            nw = torch.clamp(weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.linear(x1, w1)
                Z2 = F.linear(x2, w2)
                S1 = safe_divide(R, Z1 + Z2)
                S2 = safe_divide(R, Z1 + Z2)
                C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
                C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances

        return R



class SNConv2d(SNLinear):
        """Spectral norm based 2d conv."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: None,
            stride: None,
            #padding: t.Union[int, t.Iterable[int]] = 0,
            #dilation: t.Union[int, t.Iterable[int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",  # NB(jramapuram): not used
            init_multiplier: float = 1.0,
            stats_only: bool = False,
        ):
            #kernel_size = to_2tuple(kernel_size)
            #stride = to_2tuple(stride)
            in_features = in_channels * kernel_size[0] * kernel_size[1]
            super().__init__(
                in_features,
                out_channels,
                bias=bias,
                init_multiplier=init_multiplier,
                stats_only=stats_only,
            )

            assert padding_mode == "zeros"
            self.kernel_size = kernel_size
            self.stride = stride
            #self.padding = padding
            #self.groups = groups
            #self.dilation = dilation
            self.stats_only = stats_only

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            weight = self.get_weight()
            weight = weight.view(
                self.out_features, -1, self.kernel_size[0], self.kernel_size[1]
            )
            return F.conv2d(
                x,
                weight,
                bias=self.bias,
                stride=self.stride,
                #padding=self.padding,
                #dilation=self.dilation,
                #groups=self.groups,
            )
        

