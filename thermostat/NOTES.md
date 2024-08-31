How do you enforce that the continuous-time functions are really continuous (are they required to be differentiable, etc.)

Time delta idea seems to line up nicely

Next step is probably to hook into the Unicorn logic for the controller

What needs to be white box?
- set of discrete modes D
- guard function G
- reset function R
- and therefore postDisc

What can be black box?
- Flow function F
- and therefore postCont