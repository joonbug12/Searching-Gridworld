#This file is our own implementation of the Binary Heap
#Authors: Joon Song, Anay Kothana

class BinaryHeap:
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
    def contains(self, element):
        node = element[1]  # Only check for the node, not the f-value
        return any(item[1] == node for item in self.heap)

    #used for when inserting a node into the heap. If the index being inserted has a higher priority than the parent,
    #it will swim up the heap. Process repeats until the node is in the correct position.
    def swim(self, index):
        parentIndex = self.parent(index)

        if index>0 and self.heap[index] < self.heap[parentIndex]:
            #swap if the current element has a higher priority than the parent
            self.heap[index], self.heap[parentIndex] = self.heap[parentIndex], self.heap[index]
            self.swim(parentIndex)

    #this method ensures that the heap is properly maintained after an element is taken out
    def sink(self, index):
        leftIndex = self.left_child(index)
        rightIndex = self.right_child(index)
        smallest = index

        if leftIndex<len(self.heap) and self.heap[leftIndex]<self.heap[smallest]:
            smallest = leftIndex
        
        if rightIndex<len(self.heap) and self.heap[rightIndex]<self.heap[smallest]:
            smallest = rightIndex
        
        if smallest != index:
            #swap with the smaller child and continue
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

    
