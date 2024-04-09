Filename = ""


class LogStore():
    def __init__(self):
        self.filename = ""

    def setfilename(self,name):
        self.filename = name
    
    def getfilename(self):
        return self.filename

    def write_to_log(self, input):
            with open("code/" +self.filename + ".txt", 'a') as log:
                log.write(input + "\n")

log = LogStore()