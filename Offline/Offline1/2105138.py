
def polynomial_multiplication(poly1, poly2):
    result = [0] * (len(poly1) + len(poly2) - 1)
    
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            result[i + j] += poly1[i] * poly2[j]
    
    return result

def main():

    d1 = int(input("Degree of the first Polynomial: "))
    poly1 = list(map(int, input("Coefficients: ").split()))
    
    d2 = int(input("Degree of the second Polynomial: "))
    poly2 = list(map(int, input("Coefficients: ").split()))
    
    result = polynomial_multiplication(poly1, poly2)
    print("Degree of the Polynomial:", len(result) - 1)
    print("Coefficients:", ' '.join(map(str, result)))

if __name__ == "__main__":
    main()
