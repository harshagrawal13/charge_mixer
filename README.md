## Installation
Create a new environment and install the following: 
`pip install scipy`
`pip install pandas`

## Overall Guide
The user will only enter 1 input json file that will have the compositions + cost of the raw materials available along with the the composition of the final output required.

There are three modes:
- `vanilla_optimization`: This will optimize from scratch and should show the best savings. 
- `with_existing_no_weight_constraints`: This will optimize considering only the raw materials for which the weight (substandard) is provided by the user. It won't consider raw materials for which the user doesn't provide the weight. This would show lesser savings than vanilla optimization.
- `with_existing_with_weight_constraints`: This will optimize (on top of the previous mode) with an additional min-max constraint on the raw material weights that the user wants to use. This will show the least savings in all three modes.


### Guide for `input.json` file
- `cost_per_ton`: Should be a continous float/integer. Don't put comma in between the numbers and don't put a string. `cost_per_ton: 26000` is allowed, `cost_per_ton: 26,000` or `cost_per_ton: "26000"` is not allowed.
- `element_percentage`: Please enter a float value. For example if after burning 1 Kg of raw material A, 100 grams of Mn is obtained (10%) then the float value should be `{"Mn": 0.1}` and not `{"Mn": 10}`. Please don't add percentage at the end. This is assuming that the recovery of that element is after slague. 
    -  Please provide the composition for all elements in all raw materials. For example, if you want to optimize for C, Si, Mn, and Fe, then all raw materials need to have the composition of all these 4 elements. The program will throw an error it doesn't find it.
- `mode`: Enter any of the 3 modes listed below, anything else will fail the program:
-  