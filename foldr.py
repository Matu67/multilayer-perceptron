# From https://eli.thegreenplace.net/2017/right-and-left-folds-primitive-recursion-patterns-in-python-and-haskell/
def foldr(func, init, seq):
    if not seq:
        return init
    else:
        return func(seq[0], foldr(func, init, seq[1:]))