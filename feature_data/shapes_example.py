"""
shapes_example.py

A simple example of tagged feature data for a series of shaped based examples. Illustrates the formatting required for
building a Multi-Class Perceptron object.

Labelled feature data for a series of objects belonging to one of the following categories: Line, Triangle, Square.
"""

# Different categories data is broken up into.
shape_classes = ["line", "triangle", "square"]

# Feature List
shape_feature_list = ["num_lines", "is_connected", "num_right_angles"]

# Feature Data - Notice that all data values must be numerical (in case of boolean values, use 0 for False, 1 for True)
shape_feature_data = [("line", { "num_lines": 1, "is_connected": 0, "num_right_angles": 0 }),
                      ("triangle", { "num_lines": 3, "is_connected": 1, "num_right_angles": 0 }),
                      ("triangle", { "num_lines": 3, "is_connected": 1, "num_right_angles": 1 }),
                      ("square", { "num_lines": 4, "is_connected": 1, "num_right_angles": 4}),
                      ("line", { "num_lines": 1, "is_connected": 0, "num_right_angles": 0 }),
                      ("triangle", { "num_lines": 3, "is_connected": 1, "num_right_angles": 0 }),
                      ("triangle", { "num_lines": 3, "is_connected": 1, "num_right_angles": 1 }),
                      ("square", { "num_lines": 4, "is_connected": 1, "num_right_angles": 4}),
                      ("line", { "num_lines": 1, "is_connected": 0, "num_right_angles": 0 }),
                      ("triangle", { "num_lines": 3, "is_connected": 1, "num_right_angles": 0 }),
                      ("triangle", { "num_lines": 3, "is_connected": 1, "num_right_angles": 1 }),
                      ("square", { "num_lines": 4, "is_connected": 1, "num_right_angles": 4}),
                      ("line", { "num_lines": 1, "is_connected": 0, "num_right_angles": 0 }),
                      ("triangle", { "num_lines": 3, "is_connected": 1, "num_right_angles": 0 }),
                      ("triangle", { "num_lines": 3, "is_connected": 1, "num_right_angles": 1 }),
                      ("square", { "num_lines": 4, "is_connected": 1, "num_right_angles": 4})]