# EM-n-body-Python
Classical electrodynamics n-body simulation in Python

## Electrodynamic Theory
In classical electrodynamics, the force on a charge $Q$ by another charge $q$ is:
```math
\vec{F} = \frac{qQ}{4 \pi \epsilon_0} \frac{r}{(\vec{r} \cdot \vec{u})^3} \Bigl\{ [(c^2 - v^2)\vec{u} + \vec{r} \times (\vec{u} \times \vec{a})] + \frac{\vec{V}}{c} \times \bigr[\hat{r} \times [(c^2 - v^2)\vec{u} + \vec{r} \times (\vec{u} \times \vec{a})]   \bigr] \Bigl\}
```

Where $\vec{r}$ is the vector pointing from $q$ to $Q$, $\vec{v}$ is the velocity of $q$, $\vec{a}$ is the acceleration of $q$, $\vec{u} = c\hat{r} - \vec{v}$,and $\vec{V}$ is the velocity of $Q$. $\vec{r}$, $\vec{u}$, $\vec{v}$, and $\vec{a}$ are all evaluated at the retarded time $t_r = t - \frac{|\vec{r}|}{c}$ since the speed of light is finite and the field takes time to propagate. Using superposition, the force on charge $Q$ from any number of other charges can be found. This equation's formatting is taken from Griffiths 4th edition *Introduction to Electrodynamics*.  

For this simulation, if the distance between charges is much less than the distance that light travels during the time step, the retarded time is ignored entirely. However, at distances greater than that, the retarded times are used. 

## WIP Functionality
Currently, the simulation only plots and makes videos in 2D using matplotlib, and only the 2D version can make visuals. 3D versions are in progess and will likely use vpython because matplotlib 3D plotting does not work well for this.

The simulations currently do not account for the force blowing up when two charges get too close to each other, but it's a work in progess.

Also, the retarded times are not accounted for yet and the simulation will only run if the retarded time is less than the time step.

Conservation laws for electrodynamics are also not accounted for yet due to the much higher complexity compared to mechanics.
