
# Scenario

Every trace is from a 24-hour simulation with three meals (breakfast, lunch, and dinner). Breakfast is at t = 0 min. Lunch is always 4 hours after breakfast (at t = 240 min) and dinner is always 7 hours after lunch (at t = 660 min). An insulin dose is delivered from the pump at the same time as each meal. The files are named `trace_{init_bg}_{basal}_{bc}_{lc}_{dc}` where

init_bg = starting blood glucose (mg/dL)
basal = basal rate (u/hour)
bc = breakfast carbs (grams)
lc = lunch carbs (grams)
dc = dinner carbs (grams)

# Variables

t: time (minutes)

##  Body State

G: plasma blood glucose (mg/dL)

Gp: plasma blood glucose ()

##  Scenario parameters

D_1: carboydrates in breakfast (mg)
t_1: breakfast time (min)
D_2: carbohydrates in lunch (mg)
t_2: lunch time (min)
D_3: carbohydrates in dinner (mg)
t_3: dinner time (min)

##  Pump internal state

pump_iob_0: IOB for first dose
pump_elapsed_0: time elapsed since first dose

pump_iob_1: IOB for second dose
pump_elapsed_1: time elapsed since second dose

pump_iob_2: IOB for third dose
pump_elapsed_2 time elapsed since third dose

pump_iob_3: IOB for fourth dose
pump_iob_3: time elapsed since fourth dose

pump_iob: pump's estimate of the amount of active insulin in the body

Notes:
- pump_iob_0, ..., pump_iob_3 don't decay with time (they are equal to the initial dose delivered by the pump), but pump_iob decays
- the pump can support more than four insulin doses for IOB modeling, but we only have 4 because our scenarios only have three doses 