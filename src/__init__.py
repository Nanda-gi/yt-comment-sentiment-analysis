import yaml
with open('params.yaml', 'r') as file:
    data = yaml.safe_load(file)
    print(data)