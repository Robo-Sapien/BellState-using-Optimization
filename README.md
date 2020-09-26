# Bell State Using Optimization

## Problem Statement
Implement a circuit that returns |01⟩ and |10⟩ with equal probability (50% for each).
Requirements :
- The circuit should consist only of CNOTs, RXs and RYs. :heavy_check_mark:
- Start from all parameters in parametric gates being equal to 0 or randomly chosen. :heavy_check_mark:
- You should find the right set of parameters using gradient descent (you can use more - advanced optimization methods if you like). :heavy_check_mark:
- Simulations must be done with sampling (i.e. a limited number of measurements per iteration) and noise. :heavy_check_mark:

##### Bonus question
How to make sure you produce state  |01⟩  +  |10⟩  and not any other combination of |01⟩ + ![formula](https://render.githubusercontent.com/render/math?math=e^{i%20\phi})|10⟩ 
(for example |01⟩ - |10⟩)? :heavy_check_mark:

## Solution
`Note: Multiple solutions exist for this problem & bonus question`
### Circuit
Initial state to the circuit is |00⟩. I have used 2 RY gates and 1 CNOT gate.

![Optional Text](../master/plots/circuit.png)
  
Q. *Why did I measure the qubits? Wouldn't that collapse the final statevector? How would I write the loss function without the statevector?*
  
A. The task mentioned specifically to perform simulation via sampling *implying* measurements are necessary.
Yes it would collapse the statevector, hence I had to come up with a different approach to write a good loss function that incorporates the sampling effect. 



### Bonus Question
Q. *How did I ensure no relative phase between |01⟩ & |10⟩ ?*

A. I ensured no relative by these restrictions (**Bloch sphere** offers great visualization for these):
- **Don't use RX gates**: RX gates tend to bring the qubit vector outside x-z plane *which contributes to a non zero azimuthal angle, ![formula](https://render.githubusercontent.com/render/math?math=\phi)*. Hence it leads to relative phase = ![formula](https://render.githubusercontent.com/render/math?math=e^{i%20\phi}). Hence our objective is to keep the qubit vector in x-z plane.
- **Bound the RY parameters in (0, ![formula](https://render.githubusercontent.com/render/math?math=\pi))**: Since we use only RY gates, there is a finite chance that relative phase = -1 appears if vector is close to the negative x-axis.

**Note:** These gate restrictions to avoid relative phase work only if we start at |0⟩ qubit, which is true in our task.

### Optimization details
The optimizer used is COBYLA. I have written two cost functions: (1) Squared error loss (2) Entropy loss. I am using Entropy loss as it gives slightly better results. 

Q. *Why is entropy loss giving better results than squared error loss?*

A. The target is a probabilities vector, hence the optimization is closer to multi-label classification than regression. Hence a entropy loss performs better than a squared error loss. If the target would have been the statevector instead, I would have used the squared error loss. 


#### Loss plot

![Optional Text](../master/plots/LossCovergencePlot.png)

Lets see how the probabilities vary with each iteration for different number of `shots`:

- `shots` = 1 [Since we are measuring only once per iteration, we see only the 1 collapsed state]

![Optional Text](../master/OptimizationGifs/Gifshots1.gif)

- `shots` = 10 [Can see significant probabilities in |00⟩ & |11⟩ states as well]

![Optional Text](../master/OptimizationGifs/Gifshots10.gif)

- `shots` = 100 [Big improvement from previous case, insignificant probabilities in |00⟩ & |11⟩ states]

![Optional Text](../master/OptimizationGifs/Gifshots100.gif)

- `shots` = 1000 [As the shots increase, we get better results]

![Optional Text](../master/OptimizationGifs/Gifshots1000.gif)

## Results

Lets compare the results obtained for different number of `shots`:

- `shots` = 1

Training Time: 0.11134672164916992

Optimized paramters: [1.14449752 3.17674159]

Final Statevector: [-0.01477384+0.j  0.54144029+0.j  0.84055552+0.j -0.00951651+0.j]

Final loss: 13.815507557965773


![Optional Text](../master/plots/histogram_shots1.png)


- `shots` = 10

Training Time: 0.3823280334472656

Optimized paramters: [1.04222172 3.19447852]

Final Statevector: [-0.0229304 +0.j  0.49766981+0.j  0.86696349+0.j -0.01316292+0.j]

Final loss: 1.8325732137625914


![Optional Text](../master/plots/histogram_shots10.png)

- `shots` = 100

Training Time: 0.4119124412536621

Optimized paramters: [1.70888822 3.03620942]

Final Statevector: [0.0345833 +0.j 0.75315927+0.j 0.65572653+0.j 0.03972195+0.j]

Final loss: 1.3898948422670188


![Optional Text](../master/plots/histogram_shots100.png)

- `shots` = 1000

Training Time: 0.5514860153198242

Optimized paramters: [1.61166117 3.1334394 ]

Final Statevector: [0.00282311+0.j 0.72140009+0.j 0.69250653+0.j 0.0029409 +0.j]

Final loss: 1.3866884395455952

![Optional Text](../master/plots/histogram_shots1000.png)


### Library & framework versions
- Python: 3.7.7
- matplotlib: 3.2.0
- qiskit-terra: 0.15.2
- qiskit-aer: 0.6.1
- qiskit-ignis: 0.4.0
- qiskit-ibmq-provider: 0.9.0
- qiskit-aqua: 0.7.5
- qiskit: 0.21.0

