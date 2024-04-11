
# Action Space

from the project description given:
"Please observe that the actions are given as “deltas” and not as absolute angles for the joints, that means that the Agent must decide if the angles increase, decrease or keep a joint angle. However, that must be defined for all six joints, meaning that the Agent can take at every Time Stamp $3^6$ possible Actions (= 729)! Every joint angle is bounded to $-180°<θ_i<+180°  ∀i∈[1,2,3,4,5,6]$."

Action pro joint δθ_i [Degree]

- Decrease: - 0.1°

- Keep: 0.0°

- Increase: + 0.1°

## Action space definition in code

discrete nature of actions + their specific impact on the joints
--->  best using a MultiDiscrete action space in Gymnasium

our case:
gym.spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])

For example, an action of [2, 0, 1, 2, 1, 0] would indicate to increase the first joint by +0.1°, decrease the second joint by -0.1°, keep the third and fifth joints' angles the same, increase the fourth joint by +0.1°, and decrease the sixth joint by -0.1°.
