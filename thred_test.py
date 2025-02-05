import threading

class CustomThread(threading.Thread):
    def __init__(self, ):
        threading.Thread.__init__(self)
        self.value = 0
        
    def mess(self, my_mess):
        self.value = my_mess
        
    def run(self):
        print("This is my custom run!", self.value)

custom_thread = CustomThread()
custom_thread.mess("Hej")
custom_thread.start()