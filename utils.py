

def combined_shape(dim1, *args):
    if isinstance(dim1, tuple):
        return (*dim1, *args)
    return (dim1, *args)
