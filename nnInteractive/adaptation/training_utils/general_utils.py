import inspect 

def validate_named_args(cls, args):
    """Validate a dictionary of args: names -> values against cls.__init__ named
    parameters.
    """
    if args is None:
        raise Exception('args is None')
    if not isinstance(args, dict):
        raise TypeError("positional args must be a list/tuple or None")

    sig = inspect.signature(cls.__init__)
    # ordered parameters excluding 'self'
    required = {
        name for name, p in sig.parameters.items()
        if p.kind ==inspect.Parameter.POSITIONAL_OR_KEYWORD and name != "self"
    }
    # collect
    if not all(param in args for param in required):
        missing = required - set(args.keys())
        raise TypeError(f"missing required named args: {missing}")
    return args
def make_factory(cls):
    def factory(spec):
        payload = validate_named_args(cls, spec)  # raises on mismatch
        return cls(**payload) #Pass it through as kwargs to eliminate
    #manually writing every variable out for all transforms we want to use.
    return factory

