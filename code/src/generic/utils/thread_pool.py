
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool


# CPU/GPU option
# h5 requires a Tread pool while raw images requires to create new process

def create_cpu_pool(no_thread, use_process, maxtasksperchild=1000):

    if no_thread == 0:
        return None
    elif use_process:
        cpu_pool = Pool(no_thread, maxtasksperchild=maxtasksperchild)
    else:
        cpu_pool = ThreadPool(no_thread)
        cpu_pool._maxtasksperchild = maxtasksperchild

    return cpu_pool