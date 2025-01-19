import RPi.GPIO as GPIO
import time
import random


#Morse down when out_state == False

out_pin = 12
out_state = True

GPIO.setmode(GPIO.BOARD)
GPIO.setup(out_pin, GPIO.OUT)
GPIO.output(out_pin, out_state)

time.sleep(1)

while True:
    GPIO.output(out_pin, out_state)
    out_state = not out_state
    time.sleep(random.uniform(0.01, 0.25))

GPIO.cleanup()
