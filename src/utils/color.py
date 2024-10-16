


def red_str(string:str) -> str:
    return f'\x1b[1;31;40m{string}\x1b[0m'


def green_str(string:str) -> str:
    return f'\x1b[1;32;40m{string}\x1b[0m'


def gray_str(string:str) -> str:
    return f'\x1b[1;30;40m{string}\x1b[0m'