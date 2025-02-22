def to_list(arg):
    if isinstance(arg, str) or not hasattr(arg, '__iter__'):
        return [arg]
    return list(arg)
