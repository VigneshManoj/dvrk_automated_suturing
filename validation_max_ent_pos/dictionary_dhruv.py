#!/bin/bash/python

import numpy as np
import csv
import math
import pprint
pp = pprint.PrettyPrinter(indent=2)

ls_min = 0
ls_max = 1
ls_steps = 10
ls = np.linspace(ls_min, ls_max, ls_steps)

X = {round(ival, 3): int(i) for i, ival in enumerate(ls)}
Y = {round(ival, 3): int(i) for i, ival in enumerate(ls)}
Z = {round(ival, 3): int(i) for i, ival in enumerate(ls)}
# TH = {round(ival, 3): int(i) for i, ival in enumerate(ls)}
# PH = {round(ival, 3): int(i) for i, ival in enumerate(ls)}
# SC = {round(ival, 3): int(i) for i, ival in enumerate(ls)}

States = {i: (0, 0, 0, 0, 0, 0) for i in range(pow(len(ls), 6))}
for i, ival in enumerate(ls):
  for j, jval in enumerate(ls):
    for k, kval in enumerate(ls):
      # for l, lval in enumerate(ls):
      #   for m, mval in enumerate(ls):
      #     for n, nval in enumerate(ls):
        '''
        ele = {
          (int(i),
           int(j),
           int(k),
           int(l),
           int(m),
           int(n)):
            (round(ival, 3), round(jval, 3), round(kval, 3), round(lval, 3), round(mval, 3), round(nval, 3))
        }
        '''
        ele = {
            (int(i),
             int(j),
             int(k)):
                (round(ival, 3), round(jval, 3), round(kval, 3))
             }

        #yapf: disable
#         _key = int(i) * pow(len(ls), 5) + int(j) * pow(len(ls), 4) + int(k) * pow(len(ls), 3) + int(l)*pow(len(ls),2) + int(m)*pow(len(ls),1) + int(n)
        _key = int(i) * pow(len(ls), 2) + int(j) * pow(len(ls), 1) + int(k)
        del States[_key]
        ##yapf: enable
        States.update(ele)


def return_val(*argv):
    if len(argv) > 1:
    # return States[(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5])]
        return States[(argv[0], argv[1], argv[2])]

    else:
        return States[argv[0]]


def return_key(*argv):
    tup = ()
    if len(argv) == 1:
        tup = argv[0]
    else:
        tup = (argv[0], argv[1], argv[2])
        # tup = (argv[0], argv[1], argv[2], argv[3], argv[4], argv[5])

    # return (X[tup[0]], Y[tup[1]], Z[tup[2]], TH[tup[3]], PH[tup[4]], SC[tup[5]])
    return (X[tup[0]], Y[tup[1]], Z[tup[2]])


print 'Checking: 0.0, 1.0, 0.889'
chk = (0.0, 1.0, 0.889)
print return_key(chk)
print return_val(return_key(chk))