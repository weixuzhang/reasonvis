# import openai
# from openai import OpenAI
# client = OpenAI(api_key = "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")

# response = client.chat.completions.create(
#   model="gpt-4",
#   messages=[
#     {
#       "role": "system",
#       "content": "Given the following SQL tables, your job is to write queries given a user’s request.\n    \n    CREATE TABLE Orders (\n      OrderID int,\n      CustomerID int,\n      OrderDate datetime,\n      OrderTime varchar(8),\n      PRIMARY KEY (OrderID)\n    );\n    \n    CREATE TABLE OrderDetails (\n      OrderDetailID int,\n      OrderID int,\n      ProductID int,\n      Quantity int,\n      PRIMARY KEY (OrderDetailID)\n    );\n    \n    CREATE TABLE Products (\n      ProductID int,\n      ProductName varchar(50),\n      Category varchar(50),\n      UnitPrice decimal(10, 2),\n      Stock int,\n      PRIMARY KEY (ProductID)\n    );\n    \n    CREATE TABLE Customers (\n      CustomerID int,\n      FirstName varchar(50),\n      LastName varchar(50),\n      Email varchar(100),\n      Phone varchar(20),\n      PRIMARY KEY (CustomerID)\n    );"
#     },
#     {
#       "role": "user",
#       "content": "Write a SQL query which computes the average total order value for all orders on 2023-04-01."
#     }
#   ],
#   temperature=0,
#   max_tokens=200,
#   top_p=1
# )

# print(response.choices[0].message.content)

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def explain_optimizer_evolution():
    # Let's demonstrate the problems that AdamW solves
    
    # 1. Basic SGD problems
    def sgd_example():
        # Problem 1: Same learning rate for all parameters
        weights = torch.tensor([1.0, 0.01], requires_grad=True)
        grad = torch.tensor([0.01, 0.01])  # Same gradient
        lr = 0.1
        
        update = lr * grad
        print("\nSGD Update:")
        print(f"Parameter values: {weights.tolist()}")
        print(f"Gradients: {grad.tolist()}")
        print(f"Updates: {update.tolist()}")
        print("Problem: Same update size despite different parameter scales")
    
    # 2. Momentum solution
    def momentum_example():
        velocity = torch.zeros(2)
        beta = 0.9
        grad = torch.tensor([0.01, -0.01])
        
        # Update velocity
        velocity = beta * velocity + grad
        
        print("\nMomentum Update:")
        print(f"Gradients: {grad.tolist()}")
        print(f"Velocity: {velocity.tolist()}")
        print("Benefit: Helps escape local minima and dampens oscillations")
    
    # 3. RMSprop/Adam addition
    def rmsprop_example():
        grad = torch.tensor([0.01, 0.01])
        squared_grad = grad * grad
        v = torch.zeros(2)
        beta2 = 0.999
        
        # Update second moment
        v = beta2 * v + (1 - beta2) * squared_grad
        
        print("\nRMSprop/Adam Second Moment:")
        print(f"Gradients: {grad.tolist()}")
        print(f"Squared gradients: {squared_grad.tolist()}")
        print(f"Second moment estimate: {v.tolist()}")
        print("Benefit: Adapts learning rate per parameter")
    
    # 4. AdamW improvement
    def adamw_example():
        # Parameters
        theta = torch.tensor([1.0, 0.01], requires_grad=True)
        m = torch.zeros(2)  # First moment
        v = torch.zeros(2)  # Second moment
        beta1 = 0.9
        beta2 = 0.999
        lr = 0.001
        weight_decay = 0.01
        epsilon = 1e-8
        t = 1
        
        # Simulate one update
        grad = torch.tensor([0.01, 0.01])
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Parameter update
        update = lr * m_hat / (torch.sqrt(v_hat) + epsilon)
        theta_before_decay = theta - update
        
        # Weight decay
        theta_after_decay = theta_before_decay * (1 - lr * weight_decay)
        
        print("\nAdamW Update:")
        print(f"Original parameters: {theta.tolist()}")
        print(f"After Adam update: {theta_before_decay.tolist()}")
        print(f"After weight decay: {theta_after_decay.tolist()}")
        print("Benefit: Proper weight decay implementation")
    
    sgd_example()
    momentum_example()
    rmsprop_example()
    adamw_example()

# Demonstrate AdamW implementation
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                # Get momentum parameters
                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']
                
                # Update moments
                m.mul_(beta1).add_(p.grad.data, alpha=1-beta1)
                v.mul_(beta2).addcmul_(p.grad.data, p.grad.data, value=1-beta2)
                
                # Bias correction
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                
                # Update parameters
                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
                
                # Weight decay
                p.data.mul_(1 - lr * weight_decay)
        
        return loss

# Demonstrate usage
def demonstrate_adamw():
    # Simple model
    model = nn.Linear(2, 1)
    optimizer = AdamW(model.parameters(), lr=0.001)
    
    # Training data
    X = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    y = torch.tensor([[3.0], [6.0]])
    
    print("\nTraining Example:")
    for epoch in range(3):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.3f}")

explain_optimizer_evolution()
demonstrate_adamw()