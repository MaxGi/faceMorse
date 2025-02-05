import RPi.GPIO as GPIO
import time
import random
import threading
import queue


#Master model class
class MorseSender:
    def __init__(self, ):
        
        self.sending = False
        self.send_open = True
        self.out_mess = None
        
        self.time_unit = 0.5
        self.letter_space = 3 * self.time_unit
        self.word_space = 7 * self.time_unit
        self.dash = 3 * self.time_unit
        self.dot = 1 * self.time_unit

        self.morse_alphabet = {
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
        
        self.out_pin = 12
        out_state = True

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.out_pin, GPIO.OUT)
        GPIO.output(self.out_pin, out_state)
        
        
        self.q = queue.Queue()
        t = threading.Thread(target=self.looper)
        t.daemon = True
        t.start()
         
    def __exit__(self, exc_type, exc_value, exc_traceback):
        print('exit method called')
        GPIO.cleanup()
        
    def send(self, mess):
        self.out_mess = mess
        
    def looper(self):
        while True:
            self.sender()
            
    def sender(self):
        
        if self.out_mess == None:
            self.sending = False
            self.send_open = True
            return
        
        mess = self.out_mess
        
        print("Try send")
        try:
            mess = mess.upper()
        except:
            mess = str(int(mess[0]))
            
        self.sending = True
        self.send_open = False
            
        print("Sending message:", mess)
            
        for letter in mess:
            if letter == " ":
                print(" ")
                GPIO.output(self.out_pin, False)
                time.sleep(self.word_space)
            else:
                print(letter)
                for i in self.morse_alphabet[letter]:
                    GPIO.output(self.out_pin, True)
    #                print(i)
                    if i == 0:
                        #dot
                        time.sleep(self.dot)
                    elif i == 1:
                        #Das
                        time.sleep(self.dash)

                    GPIO.output(self.out_pin, False)
                    time.sleep(self.dot)

                GPIO.output(self.out_pin, False)
                time.sleep(self.letter_space)
        self.sending = False
        time.sleep(4)
        self.out_mess = None
        self.send_open = True
        
