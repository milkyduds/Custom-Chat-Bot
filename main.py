allergen_database = {
    'peanuts': ['peanut butter', 'peanut cookies', 'peanut soup'],
    'milk': ['cheese', 'milk chocolate', 'yogurt'],
    'eggs': ['scrambled eggs', 'omelette', 'mayonnaise'],
    'soy': ['tofu', 'soy sauce', 'soy milk'],
    'wheat': ['bread', 'pasta', 'pizza'],
}

#Function input allergy
user_allergens = []
def input_allergens():
    print("Enter your allergens")
    while True:
        allergy = input().strip().lower()
        if allergy == "x":
            break
        user_allergens.append(allergy)

def check_safety():
    food = input("Enter food to check").strip().lower()
    for allergy in user_allergens:
        if allergy in allergy_database and food in allergen_database[allergy]:
            return f"{food} is NOT safe to eat because it contains {allergen}"
        return f"{food} is safe to eat"

def main():
    input_allergens()
    result = check_safety()
    print(result)

f __name__ == "__main__":
    main()

