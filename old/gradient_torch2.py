import torch
# f = w * x
# f = 2 * x

device = torch.device('cuda')

X = torch.tensor([1,2,3,4], dtype=torch.float32,device=device)
Y = torch.tensor([2,4,6,8], dtype=torch.float32, device=device)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

print(X)

#calculate model prediction
def forward(x):
    return w * x

# calculate loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


print(f'prediction before training: f(5) = {forward(5):.3f}')

#training

learning_rate = 0.01
n_iter = 100

for epoch in range(n_iter):
    #prediction = forwardpass
    y_pred = forward(X)
    #loss
    l = loss(Y, y_pred)
    #gradients = backwardpass
    l.backward()
    #update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
    #zero gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f} ')
    
print(f'prediction after training: f(5) = {forward(5):.3f}')