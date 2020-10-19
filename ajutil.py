# -*- coding: utf-8 -*-
"""ajason08util.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15JkaK_n8fzg4_jxE05lXVHLMnjM6HquT

**Debugging**
"""

# FOR NEXT UPDATE
# (For performance measurement, time.clock() is actually preferred, since it can't be interfered with if the system clock gets messed with, but .time() does mostly accomplish the same purpose.) –
# https://stackoverflow.com/a/56655327/9356315
# timeit.default_timer is assigned to time.time() or time.clock() depending on OS.

import time
class Stopwatch:
  def __init__(self):
    self._started = time.time()
    self._last_started = self._started
    self._laps = []
    self._totaltime = "please call stop() first!"    

  def _humanizetime(self,elapsed_time):
    mins = int(elapsed_time / 60)
    secs = int(elapsed_time - (mins * 60))
    return f'{mins}m {secs}s'    

  def stop(self, v=False):
    stoptimer = time.time()
    elapsed_time = stoptimer - self._started
    self._totaltime = self._humanizetime(elapsed_time)

    # calculating lap time
    elapsed_time = stoptimer - self._last_started    
    lap = self._humanizetime(elapsed_time)
    self._laps.append(lap)    
    if v: print(lap)        
    self._last_started = time.time()
    return lap

  @staticmethod
  def current_time(time_format="%H:%M:%S", v=False):    
    now = time.time()
    now = time.strftime(time_format, time.localtime(now))
    if v: print(now)
    return now

  # read-only properties
  @property
  def laps(self):
   return self._laps
  @property
  def totaltime(self):
   return self._totaltime


class Ajfiles:
  # convert a byte amount into kilobytes, megabytes or gigabytes
  def humanize_bytes(bytes_size,round_degree=2,measure="g"):
    switcher = {
        "g": 2**30,
        "m": 2**20,
        "k": 2**10
    }
    division = switcher[measure]
    return "{} {}b".format(round(bytes_size/division, round_degree),measure)


class Ajstructures:
  # Filter a list, based on another list
  def reference_list_filter(mylist, reference):  
    index_found = list(filter(lambda x: mylist[x] in reference, range(len(mylist)))) 
    return [mylist[x] for x in index_found]  

  #  print pairs(key,values)
  def printdict(mydict, separator = "=="):
    for key in list(mydict.keys()):
      print(f'{key} {separator} {mydict[key]}')



from IPython.display import Markdown, HTML, display
def color(string,fc="yellow"):
  return f'<font color={fc}>{string}</font>'
def jprint(text):
  display(Markdown(text))

  
def inspect_obj(obj, mro=True):
  if mro: print(f'class = {type(obj)}; ancestors: {type(obj).__mro__}')
  print(type(obj))
  for att in dir(obj):
    try: print (f'{att} : {getattr(obj,att)}\n')
    except: jprint(f'{color("some exception",fc)}')

