#Same as the BinaryHeap.py but favoring lower g-values
#Authors: Joon Song, Anay Kothana

class LowerG_BinaryHeap:
    #constructor for the binary heap
    def __init__(self):
        self.heap=[]

    #this method return the index of a parent of an element/node
    def parent(self, index): return (index-1)//2
    
    #this method finds the index of the left child of a node
    def left_child(self, index): return 2 * index + 1
    
    #this method finds the index of the right child of a node
    def right_child(self, index): return 2 * index + 2

    #gets the minimum value of the heap by returning the first index aka the root
    def get_min(self):
        if len(self.heap)==0 : return None
        return self.heap[0]
    
    #gets the size of the heap
    def get_size(self) : return len(self.heap)
    
    #returns true if the size of the heap is 0, returns false otherwise
    def is_empty(self) : return len(self.heap)==0

    #checks if a node is in the heap
    def contains(self, element): return element in self.heap

    #used for when inserting a node into the heap. If the index being inserted has a higher priority than the parent,
    #it will swim up the heap. Process repeats until the node is in the correct position.
    def swim(self, index):
        parentIndex = self.parent(index)

        while index > 0 and self._compare(self.heap[index], self.heap[parentIndex]):
            self.heap[index], self.heap[parentIndex] = self.heap[parentIndex], self.heap[index]
            index = parentIndex
            parentIndex = self.parent(index)

    #this method ensures that the heap is properly maintained after an element is taken out
    def sink(self, index):
        left = self.left_child(index)
        right = self.right_child(index)
        smallest = index

        if left < len(self.heap) and self._compare(self.heap[left], self.heap[smallest]):
            smallest = left
        if right < len(self.heap) and self._compare(self.heap[right], self.heap[smallest]):
            smallest = right

        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self.sink(smallest)


    #inserting a new element in the heap
    def insert(self, element):
        self.heap.append(element)
        self.swim(len(self.heap) -1)

    #taking the highest priority element out of the heap or basically the lowest value or basically just the root
    def extract(self):
        if len(self.heap)==0: return None
        if len(self.heap)==1: return self.heap.pop()

        root=self.heap[0]
        self.heap[0] = self.heap.pop()
        self.sink(0)
        return root
    
    def _compare(self, node1, node2):
        # Compare based on f-values first
        f1, g1, node1_coords = node1
        f2, g2, node2_coords = node2

        # Tie-breaking logic: favor lower g-values (for higher g-values, flip this comparison)
        if f1 == f2:
            return g1 < g2  # Favor lower g-value nodes
        return f1 < f2

    
