allergen_database = {
    'peanuts': ['peanut butter', 'peanut cookies', 'peanut soup', 'trail mix', 'peanut brittle'],
    'milk': ['cheese', 'milk chocolate', 'yogurt', 'butter', 'cream'],
    'eggs': ['scrambled eggs', 'omelette', 'mayonnaise'],
    'soy': ['tofu', 'soy sauce', 'soy milk'],
    'wheat': ['bread', 'pasta', 'pizza'],
    'fish': ['bass', 'flounder', 'cod'],
    'shellfish': ['crab','lobster','shrimp'],
    'treenuts': ['almonds','walnuts','pecans'],
    'sesame': ['tahini', 'sushi', 'bread', 'cereal', 'granola', 'sesame oil', 'sesame paste'],
}

#Function input allergy
user_allergens = []
def input_allergens():
    print("Enter your allergens")
    while True:
        allergen = input().strip().lower()
        if allergen == "x":
            break
        user_allergens.append(allergen)

def check_safety():
    print("Enter food to check")
    while True:
        food = input().strip().lower()
        safe = True
        if food == "x":
            break
        for allergen in user_allergens:
            if allergen in allergen_database and food in allergen_database[allergen]:
                print(f"{food} is NOT safe to eat because it contains {allergen}")
                safe = False
                break
        if safe:
            print(f"{food} is safe to eat")

def main():
    input_allergens()
    check_safety()

if __name__ == "__main__":
    main()

