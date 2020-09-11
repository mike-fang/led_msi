import sys
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

args = {"-setup": 15, "-sleep": 0.01, "-show": True, "-w": 1, "-path": os.path.join(curr_dir, 'imgs'), "-test": False}

i = 1
while i < len(sys.argv):
    args[sys.argv[i]] = sys.argv[i + 1]
    i += 2

print(args)