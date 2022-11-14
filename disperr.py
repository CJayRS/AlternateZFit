# Andreas Juettner

import numpy as np
def disperr(val,dval):
    '''
    nicely formatted value(error)
    '''
    n=len(val)
    if n!=len(dval):
        print("val and dval must have the same length!")
        print(val,dval)
        print("exiting")
        exit()
    dig=2
    res = n*['']
    for i in range(n):
        if dval[i] == 0. and val[i] == 0.:
            res[i]     = "0"
        elif np.isnan(val[i]) or np.isnan(dval[i]):
            res[i]     = "nan"
        elif dval[i] == 0. and val[i] != 0.:
            value      = "%d" % val[i]
            res[i]     = value
        elif dval[i] < 1: 
            location   = int(np.floor(np.log10(dval[i])))
            append_err="("+str(int(np.round(dval[i]*10**(-location+dig-1))))+")"
            if abs(val[i])<1e-100:
                val[i]=0.
                location=1
            valformat  = "%."+str(-location+dig-1)+"f"
            sval       = valformat % val[i]
            res[i]     = sval +append_err
        elif dval[i]>=1:
            digits     = min(0,int(np.ceil(np.log10(dval[i]))-1))+1
            error      = np.around(dval[i],digits)
            value      = np.around(val[i],digits)
            serr       = "%."+str(digits)+"f(%."+str(digits)+"f)"
            serr       = serr%(value,error)
            res[i]     = serr
        else:
            digits     = max(0,int(np.ceil(np.log10(dval[i]))-1))
            error      = int(round(dval[i]/10**digits)*10**digits)
            value      = round(val[i]/10**digits)*10**digits
            res[i]     = str(value)+"("+str(error)+")"
    return res
