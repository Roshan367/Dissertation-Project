import torch
import math
from model_architecture.attention import CustomScaledDotAttention
from model_architecture.feedforward import CustomFeedForward
from model_architecture.normalisation import NumpyNormalisation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("--- Backpropagation Checking ---")

def attention_autograd(Q, K, V):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    max_scores = scores.max(dim=-1, keepdim=True).values
    scores = scores - max_scores
    exp_scores = torch.exp(scores)
    P = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
    return torch.matmul(P, V)
    
        
def normalisation_autograd(ctx, x, gamma, beta, epsilon):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        std = torch.sqrt(var + epsilon)
        xhat = (x - mean) / std

        y = gamma * xhat + beta

        ctx.save_for_backward(
            x,
            gamma,
            beta,
        )
        ctx.xhat = xhat
        ctx.std = std
        ctx.epsilon = epsilon
        print(f"Normalisation shape: {y.shape}")
        return y

torch.manual_seed(42)
Q = torch.randn(1, 4, 8, 16, dtype=torch.float64, requires_grad=True).to(device)
K = torch.randn(1, 4, 8, 16, dtype=torch.float64, requires_grad=True).to(device)
V = torch.randn(1, 4, 8, 16, dtype=torch.float64, requires_grad=True).to(device)

Q2 = Q.clone().detach().requires_grad_(True)
K2 = K.clone().detach().requires_grad_(True)
V2 = V.clone().detach().requires_grad_(True)

out_auto = attention_autograd(Q, K, V)
loss_auto = out_auto.sum()
loss_auto.backward()

out_custom = CustomScaledDotAttention.apply(Q2, K2, V2, None, 16)
loss_custom = out_custom.sum()
loss_custom.backward()

print(f"Q gradient match: {torch.allclose(Q.grad, Q2.grad, atol=1e-4)}")
print(f"K gradient match: {torch.allclose(K.grad, K2.grad, atol=1e-4)}")
print(f"V gradient match: {torch.allclose(V.grad, V2.grad, atol=1e-4)}")

print(f"Q max gradient difference: {(Q.grad - Q2.grad).abs().max():.2e}")
print(f"K max gradient difference: {(K.grad - K2.grad).abs().max():.2e}")
print(f"V max gradient difference: {(V.grad - V2.grad).abs().max():.2e}")


def feedforward_autograd(x, W1, b1, W2, b2):
    z1 = torch.matmul(x, W1) + b1
    a1 = torch.maximum(z1, torch.tensor(0.0, dtype=z1.dtype))
    y = torch.matmul(a1, W2) + b2
    return y

torch.manual_seed(42)
x_ff = torch.randn(2, 8, 512, dtype=torch.float64, requires_grad=True)
W1 = torch.randn(512, 2048, dtype=torch.float64, requires_grad=True)
b1 = torch.randn(2048, dtype=torch.float64, requires_grad=True)
W2 = torch.randn(2048, 512, dtype=torch.float64, requires_grad=True)
b2 = torch.randn(512, dtype=torch.float64, requires_grad=True)

x_ff2 = x_ff.clone().detach().requires_grad_(True)
W1_2 = W1.clone().detach().requires_grad_(True)
b1_2 = b1.clone().detach().requires_grad_(True)
W2_2 = W2.clone().detach().requires_grad_(True)
b2_2 = b2.clone().detach().requires_grad_(True)

out_ff_auto = feedforward_autograd(x_ff, W1, b1, W2, b2)
out_ff_auto.sum().backward()

out_ff_custom = CustomFeedForward.apply(x_ff2, W1_2, b1_2, W2_2, b2_2)
out_ff_custom.sum().backward()

print("\nFeedforward:")
print(f"  x gradient match:  {torch.allclose(x_ff.grad, x_ff2.grad, atol=1e-4)}")
print(f"  W1 gradient match: {torch.allclose(W1.grad, W1_2.grad, atol=1e-4)}")
print(f"  b1 gradient match: {torch.allclose(b1.grad, b1_2.grad, atol=1e-4)}")
print(f"  W2 gradient match: {torch.allclose(W2.grad, W2_2.grad, atol=1e-4)}")
print(f"  b2 gradient match: {torch.allclose(b2.grad, b2_2.grad, atol=1e-4)}")
print(f"  x max difference:  {(x_ff.grad - x_ff2.grad).abs().max():.2e}")
print(f"  W1 max difference: {(W1.grad - W1_2.grad).abs().max():.2e}")
print(f"  W2 max difference: {(W2.grad - W2_2.grad).abs().max():.2e}")

def normalisation_autograd(x, gamma, beta, epsilon=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + epsilon)
    xhat = (x - mean) / std
    return gamma * xhat + beta

torch.manual_seed(42)
x_norm = torch.randn(2, 8, 512, dtype=torch.float64, requires_grad=True)
gamma = torch.ones(512, dtype=torch.float64, requires_grad=True)
beta = torch.zeros(512, dtype=torch.float64, requires_grad=True)

x_norm2 = x_norm.clone().detach().requires_grad_(True)
gamma2 = gamma.clone().detach().requires_grad_(True)
beta2 = beta.clone().detach().requires_grad_(True)

out_norm_auto = normalisation_autograd(x_norm, gamma, beta)
out_norm_auto.sum().backward()

out_norm_custom = NumpyNormalisation.apply(x_norm2, gamma2, beta2, 1e-5)
out_norm_custom.sum().backward()

print("\nNormalisation:")
print(f"  x gradient match:     {torch.allclose(x_norm.grad, x_norm2.grad, atol=1e-4)}")
print(f"  gamma gradient match: {torch.allclose(gamma.grad, gamma2.grad, atol=1e-4)}")
print(f"  beta gradient match:  {torch.allclose(beta.grad, beta2.grad, atol=1e-4)}")
print(f"  x max difference:     {(x_norm.grad - x_norm2.grad).abs().max():.2e}")
print(f"  gamma max difference: {(gamma.grad - gamma2.grad).abs().max():.2e}")
print(f"  beta max difference:  {(beta.grad - beta2.grad).abs().max():.2e}")