import time

class TicToc(object):
    def __init__(self, name='TicToc'):
        self.name = name
        self.running = False
        self.totaltime = 0
        self.tic_time = None
        self.toc_time = None

    def tic(self):
        if self.running == False:
            self.running = True
            self.tic_time = time.time()
        else:
            print("already running")

    def toc(self):
        if self.running == True:
            self.running = False
            self.toc_time = time.time()

            self.totaltime += self.toc_time - self.tic_time
        else:
            print("not running")
    
    def __str__(self):
        local_h, local_m, local_s = self.__split_time__(self.totaltime)
        return ("%s:\t %d:%02d:%02.3f" % (self.name, local_h, local_m, local_s)) 
    
    def __split_time__(self, local_diff):
        local_h = local_diff // (60 * 60)
        local_m = (local_diff - local_h * 60 * 60) // 60
        local_s = (local_diff - local_h * 60 * 60) - local_m * 60
        return [local_h, local_m, local_s]

if __name__ == "__main__":
    tt1 = TicToc("TT1")
    tt1.tic()
    time.sleep(1)
    tt1.toc()
    tt1.tic()
    time.sleep(1)
    tt1.toc()
    print(tt1)