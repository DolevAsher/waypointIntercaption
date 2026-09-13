This is a Genetic Algorithm project!

This project uses a genetic algorithm to teach pilot bots how to intercept waypoints.
In a flight simulator environment we initialize one-hundred bots.
Each bot is represented by a set of weights over a fully connected feed forward neural network.
The FFN gets the next inputs:
1. 
2. 
3.
...

And outputs the next flying orders:
1. Roll
2. Pitch

The form of the NN:
(a picture of the connections)

We run the simulation and choose the best bots out of the candidates.
Then we multiply the best and make minor changes to create the **evolution effect**.
After a few dosens of generations there is a visible improvement.
