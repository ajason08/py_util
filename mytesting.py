from ajason08util import Stopwatch
import time

def testing_stopwatch():
  mywatch = Stopwatch()
  time.sleep(2)
  mywatch.stop()
  print(mywatch.laps)
  time.sleep(3)
  mywatch.stop()
  print(mywatch.laps)
  print(mywatch.totaltime)
  print("current", mywatch.current_time())
#testing_stopwatch()

mywatch = Stopwatch()
#print(f'lol:{mywatch.stop()}')
Stopwatch().current_time(time_format="%H:%M:%S", v=True)