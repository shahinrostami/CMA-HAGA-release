import ctypes
import numpy as np

def worst_chv(population, ref_point, nobj, popsize):
    front = population

    f = ctypes.CDLL('IWFG/iwfg.so').main_f
    f.restype = ctypes.c_int
    f.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double))

    c_double_p = ctypes.POINTER(ctypes.c_double)
    data = front.flatten('F')
    data = data.astype(np.double)
    data_p = data.ctypes.data_as(c_double_p)

    refs = ref_point
    refs = refs.astype(np.double)
    refs_p = refs.ctypes.data_as(c_double_p)

    result = f(popsize, nobj, data_p, refs_p)
    return result

def chv_selection(population, ref_point, cap):
    
    front = population[:]
    popsize = front.shape[0]
     
    nobj = front.shape[1]

    worst_idx = np.empty(popsize - cap)
    
    for x in range(0,popsize - cap):
        rejected_idx = worst_chv(front, ref_point, nobj, popsize-x)
        
        rejected_objv = front[rejected_idx,:]
        
        worse = np.where(np.all(population==rejected_objv,axis=1))
        front = np.delete(front, (rejected_idx), axis=0)
        
        worst_idx[x] = worse[0][0]      
        
    return worst_idx.astype(int)