from micrograd import Value 
from nn import MLP 
import random

random.seed(42)

model = MLP(2, [3, 4, 1])
# dataset
xs = [
    [Value(2.0),  Value(3.0)],
    [Value(-1.0), Value(1.0)],
    [Value(0.5),  Value(1.0)],
    [Value(1.0),  Value(-1.0)],
]
ys = [Value(1.0), Value(-1.0), Value(-1.0), Value(1.0)]  # targets

for step in range(100): 
    
    # 1. forward pass 
    ypred = [model(x) for x in xs] 
    loss = sum((yout - ygt)**2 for yout, ygt in zip(ypred, ys)) 

    #2. zero gradients 
    for p in model.parameters():  
        p.grad = 0 

    # 3. backward pass 
    loss.backward() 

    # 4. update weights 
    for p in model.parameters(): 
        p.data -= 0.01 * p.grad 


    print(f'step {step}, loss {loss.data:.4f}')

print('\nPredictions vs targets:')
for x, y in zip(xs, ys):
    pred = model(x)
    print(f'target: {y.data:.1f}, predicted: {pred.data:.4f}')  