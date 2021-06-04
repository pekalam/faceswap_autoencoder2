import os

def find_dataset_folder(folder_path: str) -> str:
    if os.path.exists(folder_path):
        return folder_path
    if folder_path.startswith('..'):
        folder_path = folder_path.replace('..' + os.path.sep, '')
    to_find = folder_path.split(os.path.sep)[0]
    if to_find == '..':
        raise ValueError("Path cannot contain ..")
    search_lvl = []
    cwd = os.getcwd()
    max_lvl = len(cwd.split(os.path.sep))
    print(f'searching in {cwd} and {max_lvl} parent paths')
    while True:
        folder_path = os.path.join('..', folder_path)
        search_lvl.append('..')
        if len(search_lvl) == max_lvl:
            raise ValueError("Cannot find " + to_find)
        search_path = os.path.sep.join(search_lvl)
        print('searching in: ', search_path)
        for f in os.listdir(search_path):
            if to_find in f:
                print("found in ", folder_path)
                return folder_path
