# -*- coding: utf-8 -*-
"""Untitled61.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GXoh2ZRaD0MUfr8AJpZl9JTwCj--GpM5
"""

# Assignment 2 – Singly Linked List using OOP
# Author: Ravish Kumar

class Node:
    """Represents a single node in a singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """Manages the singly linked list."""
    def __init__(self):
        self.head = None

    def add_node(self, data):
        """Add a node with given data to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        temp = self.head
        while temp.next:
            temp = temp.next
        temp.next = new_node

    def print_list(self):
        """Print all elements in the list."""
        if not self.head:
            print("The list is empty.")
            return
        temp = self.head
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")

    def delete_nth_node(self, n):
        """Delete the nth node (1-based index) from the list."""
        if not self.head:
            raise Exception("Cannot delete from an empty list.")

        if n <= 0:
            raise IndexError("Index must be a positive integer.")

        if n == 1:
            self.head = self.head.next
            return

        temp = self.head
        count = 1

        while temp and count < n - 1:
            temp = temp.next
            count += 1

        if not temp or not temp.next:
            raise IndexError("Index out of range.")

        temp.next = temp.next.next


# ----------------------------
# 🧪 Test the LinkedList
# ----------------------------

if __name__ == "__main__":
    print("✅ Creating Linked List:")
    ll = LinkedList()

    ll.add_node(10)
    ll.add_node(20)
    ll.add_node(30)
    ll.add_node(40)

    print("🔁 Initial List:")
    ll.print_list()

    print("\n❌ Deleting 3rd node...")
    try:
        ll.delete_nth_node(3)
        ll.print_list()
    except Exception as e:
        print("Error:", e)

    print("\n❌ Trying to delete 10th node (out of range)...")
    try:
        ll.delete_nth_node(10)
    except Exception as e:
        print("Error:", e)

    print("\n❌ Trying to delete from empty list...")
    try:
        empty_list = LinkedList()
        empty_list.delete_nth_node(1)
    except Exception as e:
        print("Error:", e)