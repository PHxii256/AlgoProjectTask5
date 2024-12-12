class BTreeNode:
    def __init__(self, t, leaf=False):
        # Minimum degree (t) and leaf flag
        self.t = t
        self.leaf = leaf
        self.keys = []  # List of keys (cities)
        self.values = []  # List of values (rows)
        self.children = []  # List of children nodes
class BTree:
    def __init__(self, t):
        self.t = t  # Minimum degree (t)
        self.root = BTreeNode(t, True)
        
    def search(self, node, key):
        i = 0
        # Find the first key greater than or equal to key
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        # If key matches the node's key
        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]  # Return the corresponding row (value)
        # If this is a leaf node, return None (key not found)
        if node.leaf:
            return None
        # Recursively search in the child node
        return self.search(node.children[i], key)
    
    def insert(self, key, value):
        # If root is full, then the tree grows in height
        if len(self.root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(self.t, False)
            # Old root becomes a child of new root
            new_root.children.append(self.root)
            # Split the old root and move a key to the new root
            self.split_child(new_root, 0)
            # New root has two children, insert in the appropriate child
            self.insert_non_full(new_root, key, value)
            self.root = new_root
        else:
            self.insert_non_full(self.root, key, value)
            
    def insert_non_full(self, node, key, value):
        i = len(node.keys) - 1
        # If the node is a leaf, insert the key-value pair
        if node.leaf:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            node.keys.insert(i + 1, key)
            node.values.insert(i + 1, value)
        else:
            # Find the child that should contain the key
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            # If the child is full, split it
            if len(node.children[i].keys) == 2 * self.t - 1:
                self.split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            # Insert the key into the appropriate child
            self.insert_non_full(node.children[i], key, value)
            
    def split_child(self, parent, index):
        t = self.t
        node = parent.children[index]
        new_node = BTreeNode(t, node.leaf)
        parent.keys.insert(index, node.keys[t - 1])
        parent.values.insert(index, node.values[t - 1])
        parent.children.insert(index + 1, new_node)
        new_node.keys = node.keys[t:]
        new_node.values = node.values[t:]
        node.keys = node.keys[:t - 1]
        node.values = node.values[:t - 1]
        if not node.leaf:
            new_node.children = node.children[t:]
            node.children = node.children[:t]
            
    def print_tree(self, node, level=0):
        # Helper method to print the B-tree structure
        print('Level', level, 'Keys:', node.keys)
        if not node.leaf:
            for child in node.children:
                self.print_tree(child, level + 1)
