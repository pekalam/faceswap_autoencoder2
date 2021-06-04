

def get_array_from_params_split(array_str: str):
    if array_str.startswith('['):
        return eval(array_str)
    else:
        return array_str