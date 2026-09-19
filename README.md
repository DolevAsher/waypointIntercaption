## > WAYPOINT_INTERCEPTION

This is a **Genetic Algorithm project** that teaches pilot bots how to
intercept waypoints in a flight-simulator environment.

### 🧬 Genetic Algorithm

We initialize **100 pilot bots**. Each bot is represented by a set of
weights for a fully connected feed-forward neural network.

The neural network receives the following inputs:

1. Alignment cosine — angle between the heading and target direction
2. Azimuth error — horizontal steering error
3. Elevation error — vertical steering error
4. Distance to target waypoint
5. Current pitch
6. Current roll
7. Current altitude

The network produces the next flight commands:

1. Pitch
2. Roll

### 🧠 Policy Network

![PolicyNet architecture](assets/nn_diagram.svg)

```text
7 Inputs
   │
   ▼
Linear(7 → 24) + Tanh
   │
   ▼
Linear(24 → 24) + Tanh
   │
   ▼
Linear(24 → 2) + Tanh
   │
   ▼
2 Outputs
(Pitch, Roll)
```

## 🎥 Demonstration

[▶️ Watch the Waypoint Interception demonstration on YouTube](https://youtu.be/ZZO6a6ROt7A)



