import streamlit as st
import numpy as np
import math
import sympy as sp

def caesar_cipher(text, key):
    encrypted_text = ""
    for char in text:
        if char.isalpha():
            if char.isupper():
                encrypted_text += chr((ord(char) - 65 + key) % 26 + 65)
            else:
                encrypted_text += chr((ord(char) - 97 + key) % 26 + 97)
        else:
            encrypted_text += char
    return encrypted_text


#------------------------------------------------------------------------------------------------------------
#Caesar cipher decryption
def caesar_decipher(ciphertext, key):
    decrypted_text = ""
    for char in ciphertext:
        if char.isalpha():
            if char.isupper():
                decrypted_text += chr((ord(char) - 65 - key) % 26 + 65)
            else:
                decrypted_text += chr((ord(char) - 97 - key) % 26 + 97)
        else:
            decrypted_text += char
    return decrypted_text

#------------------------------------------------------------------------------------------------------------
#one time pad encryption and decryption
def Vernam(Plain,Key,Flag):
    Plain=Plain.upper()
    Key = Key.upper()
    result=""
    lengthK=len(Key)
    lengthP=len(Plain)
    for i in range(lengthP):
        char=Plain[i]
        if char.isalpha():
            if(Flag):
                result += chr((ord(char) + ord(Key[i % lengthK]) - 2 * ord('A')) % 26 + ord('A'))
            else:
                result += chr((ord(char) - ord(Key[i % lengthK]) - 2 * ord('A')) % 26 + ord('A'))

        else :
            result+=char
    return result



#------------------------------------------------------------------------------------------------------------

# playfair encryption
def prepare_input(text):
    text = text.replace(" ", "").upper()
    text = text.replace("J", "I")
    return text


def generate_key(key):
    key = prepare_input(key)

    playfair_matrix = [['' for _ in range(5)] for _ in range(5)]
    key_set = set()

    i, j = 0, 0
    for letter in key:
        if letter not in key_set:
            playfair_matrix[i][j] = letter
            key_set.add(letter)
            j += 1
            if j == 5:
                i += 1
                j = 0

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for letter in alphabet:
        if letter != "J" and letter not in key_set:
            playfair_matrix[i][j] = letter
            key_set.add(letter)
            j += 1
            if j == 5:
                i += 1
                j = 0

    return playfair_matrix


def find_char_position(matrix, char):
    for i in range(5):
        for j in range(5):
            if matrix[i][j] == char:
                return i, j


def playfair_encrypt(plaintext, key):
    matrix = generate_key(key)
    plaintext = prepare_input(plaintext)
    cipher_text = ""

    i = 0
    while i < len(plaintext):
        if i == len(plaintext) - 1:
            plaintext += "X"
        elif plaintext[i] == plaintext[i + 1]:
            plaintext = plaintext[:i + 1] + "X" + plaintext[i + 1:]
        i += 2

    for i in range(0, len(plaintext), 2):
        char1, char2 = plaintext[i], plaintext[i + 1]
        row1, col1 = find_char_position(matrix, char1)
        row2, col2 = find_char_position(matrix, char2)

        if row1 == row2:
            cipher_text += matrix[row1][(col1 + 1) % 5] + matrix[row2][(col2 + 1) % 5]
        elif col1 == col2:
            cipher_text += matrix[(row1 + 1) % 5][col1] + matrix[(row2 + 1) % 5][col2]
        else:
            cipher_text += matrix[row1][col2] + matrix[row2][col1]

    return cipher_text

#------------------------------------------------------------------------------------------------------------

#playfair decryption
def prepare_input(text):
    # Remove spaces and convert to uppercase
    text = text.replace(" ", "").upper()
    # Replace 'J' with 'I'
    text = text.replace("J", "I")
    return text


def generate_key(key):
    key = prepare_input(key)

    playfair_matrix = [['' for _ in range(5)] for _ in range(5)]
    key_set = set()

    i, j = 0, 0
    for letter in key:
        if letter not in key_set:
            playfair_matrix[i][j] = letter
            key_set.add(letter)
            j += 1
            if j == 5:
                i += 1
                j = 0

    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    for letter in alphabet:
        if letter != "J" and letter not in key_set:
            playfair_matrix[i][j] = letter
            key_set.add(letter)
            j += 1
            if j == 5:
                i += 1
                j = 0

    return playfair_matrix


def find_char_position(matrix, char):
    for i in range(5):
        for j in range(5):
            if matrix[i][j] == char:
                return i, j


def playfair_decrypt(ciphertext, key):
    matrix = generate_key(key)
    ciphertext = prepare_input(ciphertext)
    plaintext = ""

    for i in range(0, len(ciphertext), 2):
        char1, char2 = ciphertext[i], ciphertext[i + 1]
        row1, col1 = find_char_position(matrix, char1)
        row2, col2 = find_char_position(matrix, char2)

        if row1 == row2:
            plaintext += matrix[row1][(col1 - 1) % 5] + matrix[row2][(col2 - 1) % 5]
        elif col1 == col2:
            plaintext += matrix[(row1 - 1) % 5][col1] + matrix[(row2 - 1) % 5][col2]
        else:
            plaintext += matrix[row1][col2] + matrix[row2][col1]

    return plaintext


#------------------------------------------------------------------------------------------------------------


# Rail fence encryption
def encrypt_rail_fence(text: str, key: int) -> str:
    text_without_spaces = ''.join(char for char in text if char.isalnum())  # Omit spaces
    rail: list[str] = [""] * key

    dir_down = False
    row = 0

    for i in range(len(text_without_spaces)):
        if row == 0 or row == key - 1:
            dir_down = not dir_down

        rail[row] += text_without_spaces[i]

        if dir_down:
            row += 1
        else:
            row -= 1

    return "".join(rail)
#------------------------------------------------------------------------------------------------------
# Rail fence decryption
def decrypt_rail_fence(cipher: str, key: int) -> str:
    rail: list[list[str]] = [['\n' for _ in range(len(cipher))] for _ in range(key)]

    dir_down = False
    row, col = 0, 0

    for i in range(len(cipher)):
        if row == 0 or row == key - 1:
            dir_down = not dir_down

        rail[row][col] = '*'
        col += 1

        if dir_down:
            row += 1
        else:
            row -= 1

    index = 0
    for i in range(key):
        for j in range(len(cipher)):
            if rail[i][j] == '*':
                rail[i][j] = cipher[index]
                index += 1

    result = ""
    dir_down = False
    row, col = 0, 0
    for i in range(len(cipher)):
        if row == 0 or row == key - 1:
            dir_down = not dir_down

        if rail[row][col] != '*':
            result += rail[row][col]
            col += 1

        if dir_down:
            row += 1
        else:
            row -= 1

    return result


#------------------------------------------------------------------------------------------------------------
#hill encryption
def encrypt(plaintext, key):
    
    det = int(np.round(np.linalg.det(key)))  # Calculate the determinant of the key matrix
    print(det)
    # Ensure the determinant is relatively prime to 26
    if np.gcd(det, 26) != 1:
        return "Key is not invertible. Decryption not possible \n,So we not support to encrypt with this key!"
    
    plaintext = plaintext.replace (' ', '').upper()
    n = len(key)
    if len(plaintext) % n != 0:
        plaintext += 'X' * (n - (len(plaintext) % n))

    ciphertext = ''

    for i in range(0, len(plaintext), n):
        chunk = plaintext[i:i+n]
        chunk_indices = [ord(char) - ord('A') for char in chunk]
        transformed_chunk = np.dot(key, chunk_indices) % 26
        ciphertext += ''.join(chr(index + ord('A')) for index in transformed_chunk)

    return ciphertext


#------------------------------------------------------------------------------------------------------------
#hill decryption
def mod_inverse(a, m):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None

# Function to decrypt the text
def decrypt(ciphertext, key):
    # Check if the key is a square matrix
    if key.shape[0] != key.shape[1]:
        return "The shape not match"

    det = int(np.round(np.linalg.det(key)))  # Calculate the determinant of the key matrix
    print(det)
    # Ensure the determinant is relatively prime to 26
    if np.gcd(det, 26) != 1:
        return "Key is not invertible. Decryption not possible!"

    n = len(key)
    inverse_det = mod_inverse(det, 26)

    adjugate = sp.Matrix(key).adjugate()
    adjugate = np.array(adjugate.tolist(), dtype=int)

    # Perform matrix multiplication: det * adjugate
    inverse_key = (inverse_det * adjugate) % 26
    ciphertext += 'X' * (n - (len(ciphertext) % n)) 
    plaintext = ''
    for i in range(0, len(ciphertext), n):
        chunk = ciphertext[i:i+n]
        chunk_indices = [ord(char) - ord('A') for char in chunk]
        transformed_chunk = np.dot(inverse_key, chunk_indices) % 26
        plaintext += ''.join(chr(index + ord('A')) for index in transformed_chunk)

    return plaintext

#------------------------------------------------------------------------------------------------------------
#Row transposition encryption

def encryptRow(plaintext, key):
    cipher = ""
    k_indx = 0

    msg_len = float(len(plaintext))
    msg_lst = list(plaintext)
    key_lst = sorted(list(key))
    col = len(key)

    if col == 0:
        print(" ")
        return " "

    row = int(math.ceil(msg_len / col))
    fill_null = int((row * col) - msg_len)
    msg_lst.extend('x' * fill_null)
    matrix = [msg_lst[i: i + col] 
              for i in range(0, len(msg_lst), col)]

    for _ in range(col):
        curr_idx = key.index(key_lst[k_indx])
        cipher += ''.join([row[curr_idx] 
                           for row in matrix])
        k_indx += 1

    return cipher
#------------------------------------------------------------------------------------------------------------
#Row transposition decryption
def decryptRow(ciphertext, key):
    msg = ""
    k_indx = 0
    msg_indx = 0
    msg_len = float(len(ciphertext))
    msg_lst = list(ciphertext)

    col = len(key)
    if col == 0:
        print(" ")
        return " "
    
    row = int(math.ceil(msg_len / col))

    key_lst = sorted(list(key))

    dec_cipher = []
    for _ in range(row):
        dec_cipher += [[None] * col]

    for _ in range(col):
        curr_idx = key.index(key_lst[k_indx])

        for j in range(row):
            dec_cipher[j][curr_idx] = msg_lst[msg_indx]
            msg_indx += 1
        k_indx += 1

    try:
        msg = ''.join(sum(dec_cipher, []))
    except TypeError:
        raise TypeError("This program cannot",
                        "handle repeating words.")

    null_count = msg.count('_')

    if null_count > 0:
        return msg[: -null_count]

    return msg


def main():
    st.markdown("<h3 style='text-align: center;'>  Cryptography System</h3>", unsafe_allow_html=True)


    
    
    
#     f1_co,f2_co, f5_co, f6_co
    f5_co, cent_co, f4_co,  f6_co= st.columns(4)
    with cent_co:
         st.image("images/p2.jpg", width=400, caption="")
    

    
    
    algorithm_choice = st.sidebar.selectbox(
        "Choose an algorithm:",
        ["Caesar Cipher", "One Time Pad Cipher", "Playfair Cipher", "Rail Fence Cipher", "Row transposition Cipher","Hill Cipher"],
    )
    
    operation = st.radio("Choose operation:", ["Encryption", "Decryption"])
    Plain = st.text_input("Enter the text:")
    

    if algorithm_choice == "Caesar Cipher":
        key = st.number_input("Enter the key (an integer):", value=1)

        if st.button("Submit"):
            if not Plain.strip() :
                st.write("Fill in all fields first")
            else:
                if operation == "Encryption":
                    result = caesar_cipher(Plain, int(key))
                    st.write("Encrypted Result:", result)
                elif operation == "Decryption":
                    result = caesar_decipher(Plain, int(key))
                    st.write("Decrypted Result:", result)



    elif algorithm_choice == "One Time Pad Cipher":
        Key = st.text_input("Enter the key:")

        if st.button("Submit"):
            if not Plain.strip() or not Key.strip():
                st.write("Fill in all fields first")
                
            else:
                if operation == "Encryption":
                    result = Vernam(Plain, Key, True)
                    st.write("Encrypted Result:", result)
                else:
                    result = Vernam(Plain, Key, False)
                    st.write("Decrypted Result:", result)

                    
    elif algorithm_choice == "Playfair Cipher":
        key = st.text_input("Enter the key (string):")

        if st.button("Submit"):
            if not Plain.strip() or not key.strip():
                st.write("Fill in all fields first")
            else:
                if operation == "Encryption":
                    result = playfair_encrypt(Plain, key)
                    st.write("Encrypted Result:", result)
                elif operation == "Decryption":
                    result = playfair_decrypt(Plain, key)
                    st.write("Decrypted Result:", result)

                    

        

    elif algorithm_choice == "Rail Fence Cipher":
        key = st.number_input("Enter the key (int):",value=2)
        
        
        if st.button("Submit"):
           
            if not Plain.strip() :
                st.write("Fill in all fields first")
            else:
                if key > 0:
                    if operation == "Encryption":
                        result = encrypt_rail_fence(Plain, int(key))
                        st.write("Encrypted Result:", result)
                    elif operation == "Decryption":
                        result = decrypt_rail_fence(Plain, int(key))
                        st.write("Decrypted Result:", result)
                else:
                    st.write("The key must be > 0")


        
    elif algorithm_choice == "Row transposition Cipher":
        key = st.text_input("Enter the key (string):")
        
        if st.button("Submit"):
           
            if not Plain.strip() :
                st.write("Fill in all fields first")
            else:
                
                if operation == "Encryption":
                    result = encryptRow(Plain, key)
                    st.write("Encrypted Result:", result)
                elif operation == "Decryption":
                    result = decryptRow(Plain, key)
                    st.write("Decrypted Result:", result)
               
                   
        

    elif algorithm_choice == "Hill Cipher":
#         operation = st.radio("Choose operation:", ["Encryption", "Decryption"])
        order = st.number_input("Enter the order of the key:", min_value=2, step=1)
        key = []
        st.write("Enter the key:")
        for i in range(order):
            row = st.text_area(f"Enter row {i + 1} (contain {order} space-separated integers):")
            key.append(list(map(int, row.split())))

        key = np.array(key)
        
        if st.button("Submit"):
            if not Plain.strip():
                st.write("Fill in all fields first")
            else :
                if operation == "Encryption":
                    result = encrypt(Plain, key)
                    st.write("Encrypted Result:", result)
                elif operation == "Decryption":
                    result = decrypt(Plain, key)
                    st.write("Decrypted Result:", result)
                    


if __name__ == "__main__":
    main()
