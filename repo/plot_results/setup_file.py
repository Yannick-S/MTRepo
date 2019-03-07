import json

def save_file(path, experiment):
    with open(path, 'w')  as file:
        json.dump(experiment, file)

def load_file(path):
    with open(path, 'r') as file:
        json_str = file.read()
        obj = json.loads(json_str)
    return obj

if __name__ == "__main__":
    save_file('test.json', {'1':1,'2':3})
    load_file('test.json')