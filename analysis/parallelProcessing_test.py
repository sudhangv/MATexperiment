from multiprocessing import Pool

def func2(x):
    return x ** 2

def func(x, *args, **kwargs):
    return x ** 2 + sum(args)
def parallel_task():
    arg_arr = [*(1,(1,1)),2,3]
    with Pool(4) as p:
        rows = list(p.imap(func, arg_arr))
    return rows

if __name__ == '__main__':
    result = parallel_task()
    print(result)





