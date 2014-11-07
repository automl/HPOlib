import cPickle
import re
import sys

sys.stdout.write("NO WARRANTY THAT THIS METHOD WILL OPERATE WITHOUT ERROR "
                 "OR INTERRUPTION.")

print "Removing minus from %s" % sys.argv[1]
exp = cPickle.load(file(sys.argv[1]))

for t in exp['trials']:
    for p in t['params'].keys():
        new_key = re.sub('^-', '', p)
        t['params'][new_key] = t['params'][p]
        del t['params'][p]

fh = open(sys.argv[1], 'w')
cPickle.dump(exp, fh)
fh.close()