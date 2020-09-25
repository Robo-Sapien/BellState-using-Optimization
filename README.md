# Bell State Using Optimization

### Problem Statement
Implement a circuit that returns |01⟩ and |10⟩ with equal probability (50% for each).
Requirements :
- The circuit should consist only of CNOTs, RXs and RYs. 
- Start from all parameters in parametric gates being equal to 0 or randomly chosen. 	
- You should find the right set of parameters using gradient descent (you can use more - advanced optimization methods if you like). 
- Simulations must be done with sampling (i.e. a limited number of measurements per iteration) and noise.

##### Bonus question
How to make sure you produce state  |01⟩  +  |10⟩  and not any other combination of |01⟩ + ![formula](https://render.githubusercontent.com/render/math?math=e^{i%20\phi})|10⟩ 
(for example |01⟩ - |10⟩)?

### Solution
`Note: Multiple solutions exist for this problem & bonus question`
#### Circuit
Initial state to the circuit is |00⟩. I have used 2 RY gates and 1 CNOT gate.
<circuit image>
Q. *Why did I measure the qubits? Wouldn't that collapse the final statevector? How would I write the loss function without the statevector?*
A. The task mentioned specifically to perform simulation via sampling *implying* measurements are necessary.
Yes it would collapse the statevector, hence I had to come up with a different approach to write a good loss function that incorporates the sampling effect. 



#### Bonus Question
Q. *How did I ensure no relative phase between |01⟩ & |10⟩ ?*
A. I ensured no relative by these restrictions (**Bloch sphere** offers great visualization for these):
- **Don't use RX gates**: RX gates tend to bring the qubit vector outside x-z plane *which contributes to a non zero azimuthal angle, ![formula](https://render.githubusercontent.com/render/math?math=\phi)*. Hence it leads to relative phase = ![formula](https://render.githubusercontent.com/render/math?math=e^{i%20\phi}). Hence our objective is to keep the qubit vector in x-z plane.
- **Bound the RY parameters in (0, ![formula](https://render.githubusercontent.com/render/math?math=\pi))**: Since we use only RY gates, there is a finite chance that relative phase = -1 appears if vector is close to the negative x-axis.

**Note:** These gate restrictions to avoid relative phase work only if we start at |0⟩ qubit, which is true in our task.

#### Optimization details

#### Graphs & Plots
