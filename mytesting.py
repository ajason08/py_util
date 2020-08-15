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


def testing_Ajfiles():
  print(Ajfiles.humanize_bytes(545646546546,measure="k"))

def testing_Ajstructures():
  mylist = [2,4,6,8,10,12,14,16]
  reference = [6,7,8,9,16]
  print(Ajstructures.reference_list_filter(mylist,reference))

#testing_Ajstructures()

def testing_autor():
  print("f")