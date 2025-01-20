import RPi.GPIO as GPIO
import time
import random


time_unit = 0.05


letter_space = 3 * time_unit
word_space = 7 * time_unit
dash = 3 * time_unit
dot = 1 * time_unit

morse_alphabet = {
    'A': [0, 1],
    'B': [1, 0, 0, 0],
    'C': [1, 0, 1, 0],
    'D': [1, 0, 0],
    'E': [0],
    'F': [0, 0, 1, 0],
    'G': [1, 1, 0],
    'H': [0, 0, 0, 0],
    'I': [0, 0],
    'J': [0, 1, 1, 1],
    'K': [1, 0, 1],
    'L': [0, 1, 0, 0],
    'M': [1, 1],
    'N': [1, 0],
    'O': [1, 1, 1],
    'P': [0, 1, 1, 0],
    'Q': [1, 1, 0, 1],
    'R': [0, 1, 0],
    'S': [0, 0, 0],
    'T': [1],
    'U': [0, 0, 1],
    'V': [0, 0, 0, 1],
    'W': [0, 1, 1],
    'X': [1, 0, 0, 1],
    'Y': [1, 0, 1, 1],
    'Z': [1, 1, 0, 0],
    '1': [0, 1, 1, 1, 1],
    '2': [0, 0, 1, 1, 1],
    '3': [0, 0, 0, 1, 1],
    '4': [0, 0, 0, 0, 1],
    '5': [0, 0, 0, 0, 0],
    '6': [1, 0, 0, 0, 0],
    '7': [1, 1, 0, 0, 0],
    '8': [1, 1, 1, 0, 0],
    '9': [1, 1, 1, 1, 0],
    '0': [1, 1, 1, 1, 1],
    '.': [0, 1, 0, 1, 0, 1],
    ',': [1, 1, 0, 0, 1, 1]
}


test_send = "Morse code is a method of encoding text into sequences of dots and dashes. Each letter, number, and symbol has a unique combination"

#Morse down when out_state == False

out_pin = 12
out_state = True

GPIO.setmode(GPIO.BOARD)
GPIO.setup(out_pin, GPIO.OUT)
GPIO.output(out_pin, out_state)

time.sleep(1)

def sendMorse(mess):
    mess = mess.upper()
    for letter in mess:
        if letter == " ":
            print(" ")
            GPIO.output(out_pin, True)
            time.sleep(word_space)
        else:
            print(letter)
            for i in morse_alphabet[letter]:
                GPIO.output(out_pin, False)
#                print(i)
                if i == 0:
                     #dot
                     time.sleep(dot)
                elif i == 1:
                     #Das
                     time.sleep(dash)

                GPIO.output(out_pin, True)
                time.sleep(dot)

            GPIO.output(out_pin, True)
            time.sleep(letter_space)

for i in range(10):
    sendMorse(test_send)
#while True:
#    GPIO.output(out_pin, out_state)
#    out_state = not out_state
#    time.sleep(random.uniform(0.01, 0.25))

GPIO.cleanup()
