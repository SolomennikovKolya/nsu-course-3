import re
import networkx as nx
import matplotlib.pyplot as plt

with open('sem_net.txt', encoding='utf-8') as f:
    lines = f.readlines()

G = nx.DiGraph()

person_pattern = re.compile(r'^@(\w+):Человек\((.*?)\)$')
relation_pattern = re.compile(r'родств\(кто:@(\w+),кому:@(\w+)\)')

for line in lines:
    line = line.strip()

    person_match = person_pattern.match(line)
    if person_match:
        name, attrs = person_match.groups()
        G.add_node(name)

        spouse = re.search(r'супруг:@(\w+)', attrs)
        if spouse:
            G.add_edge(name, spouse.group(1), label='супруг')

        children = re.search(r'дети:\{@(.*?)\}', attrs)
        if children:
            for child in children.group(1).split(',@'):
                G.add_edge(name, child.strip(), label='родитель')

        parents = re.search(r'родители:\{@(.*?)\}', attrs)
        if parents:
            for parent in parents.group(1).split(',@'):
                G.add_edge(parent.strip(), name, label='родитель')

    relation_match = relation_pattern.match(line)
    if relation_match:
        G.add_edge(relation_match.group(1), relation_match.group(2), label='родств')

pos = nx.spring_layout(G, k=0.8)
edge_labels = nx.get_edge_attributes(G, 'label')

plt.figure(figsize=(15, 12))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title("Семантическая сеть родства")
plt.axis('off')
plt.tight_layout()
plt.show()
