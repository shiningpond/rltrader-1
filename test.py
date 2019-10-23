class A:
    def __init__(self, *args, kwarg1=1, kwarg2=2, **kwargs):
        print(args)
        print(kwargs)


if __name__ == "__main__":
    A(1,2)