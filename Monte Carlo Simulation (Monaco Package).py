#The first Algo will predict the probability of dice rolls using the "Random" library, and a regular 6-sided dice.

#The Second Algo uses MonteCarlo from the Monaco library to predict probabilities of a 20 sides die. (More applicable to Quant Finance & Common interview question)

import random

def roll_dice(num_sides=6):
    return random.randint(1, num_sides)
def monte_carlo_dice_rolls(num_rolls=1000, num_sides=6):
    rolls = [roll_dice(num_sides) for _ in range(num_rolls)]
    counts = [rolls.count(i) for i in range(1, num_sides+1)]
    total_rolls = sum(counts)
    probabilities = [count / total_rolls for count in counts]
    return probabilities
probabilities = monte_carlo_dice_rolls()
print("Probabilities of dice rolls:")
for i, prob in enumerate(probabilities):
    print(f"Rolling {i+1}: {prob:.4f}")

    #__________________________________________________________________________________________________

import montecarlo

def roll_dice(num_sides=20):
    return montecarlo.randint(1, num_sides)

def monte_carlo_dice_rolls(num_rolls=100000, num_sides=20):
    rolls = [roll_dice(num_sides) for _ in range(num_rolls)]
    counts = [rolls.count(i) for i in range(1, num_sides+1)]
    total_rolls = sum(counts)
    probabilities = [count / total_rolls for count in counts]
    return probabilities

probabilities = monte_carlo_dice_rolls()
print("Probabilities of dice rolls:")
for i, prob in enumerate(probabilities):
    print(f"Rolling {i+1}: {prob:.4f}")