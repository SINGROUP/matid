"""
This script is used to query the Bilbao Crystallographic Server for the
improper and proper rigid transformations that can be used to permute the
Wyckoff positions of atoms in a conventional cell (these are called
normalizers). This information is needed so that one common visualization can
be decided for a certain material.
"""
import pickle
import numpy as np
import urllib.request
from bs4 import BeautifulSoup


try:
    # Fetch data for each space group
    n_groups = 230
    groups = range(1, n_groups+1)
    # groups=[187]
    proper_rigid_transform_dict = {}
    improper_rigid_transform_dict = {}
    for space_group in groups:

        print(space_group)

        # Do a http get request for the data
        url = "http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-normsets?from=wycksets&gnum={}".format(space_group)
        html_raw = urllib.request.urlopen(url).read()

        # Create the soup :)
        soup = BeautifulSoup(html_raw, 'html.parser')
        tables = soup.body.center.find_all("table")

        # If only the wyckoff set table is present, skip the further
        # analysis
        if len(tables) <= 1:
            continue
        table = tables[1]
        all_rows = table.find_all("tr", recursive=False)
        rows = all_rows[2:]

        # Figure out which column in the html page has the wyckoff letters
        # wyckoff_column_number = 4
        # assert len(header) == 4
        # assert header[2].text == "Geometrical Interpretation"
        header = all_rows[0]
        titles = header("th")
        for i, title in enumerate(titles):
            title_text = title.text
            if title_text == "Transformed WP":
                wyckoff_column_number = i+1

        # Get the transformation matrix
        original_wykoffs = list("abcdefghijklmnopqrstuvwxyzA")
        for row in rows:
            tds = row.find_all("td", recursive=False)
            transform = tds[2]
            matrix = str(transform.table.tr("td")[1].text)
            matrix_rows = matrix.split("\n")
            matrix_array = []
            for matrix_row in matrix_rows:
                formatted_row = []
                elements = matrix_row.strip().split()
                for element in elements:
                    try:
                        formatted_element = float(element)
                    except ValueError:
                        num, denum = element.split("/")
                        formatted_element = float(num)/float(denum)
                    formatted_row.append(formatted_element)
                matrix_array.append(formatted_row)

            # Add the 4. row that is always the same for Euclidean
            # transformations
            matrix_array.append([0, 0, 0, 1])

            # Invert the matrix as the Bilbao Crystalographics server
            # stores the inverse matrix as seen from how they transform the
            # Wyckoff letters
            matrix_array = np.array(matrix_array)
            matrix_array = np.linalg.inv(matrix_array)

            # Divide the matrices into two categories: Proper rigid
            # transformation and inproper rigid transormations. The
            # inproper rigid transformation should also be stored because
            # they might revert back to proper rigid transformation in a
            # lower dimensional space (2D)

            # Check that the matrix represents a proper rigid
            # transformation by checking that it is orthogonal and it's
            # determinant is +1
            test_mat = matrix_array[0:3, 0:3]
            test_inv = np.linalg.inv(test_mat)
            test_trans = test_mat.T

            is_orthogonal = np.array_equal(test_inv, test_trans)
            if not is_orthogonal:
                raise Exception("Non-orthogonal transform: {}".format(test_mat))

            determinant = np.linalg.det(test_mat)
            if determinant == 1:
                dest_dict = proper_rigid_transform_dict
            elif determinant == -1:
                dest_dict = improper_rigid_transform_dict
            else:
                raise Exception("Invalid determinant.")

            # Get the permutation pairs
            wyckoffs = tds[wyckoff_column_number].string
            transformed_letters = wyckoffs.split()
            permutations = {}
            for i, letter in enumerate(transformed_letters):
                original_letter = original_wykoffs[i]
                permutations[original_letter] = letter

            if space_group not in dest_dict:
                dest_dict[space_group] = []

            dest_dict[space_group].append({"transformation": matrix_array, "permutations": permutations})

    # Create a pickle file of the translations for later use
    pickle.dump( improper_rigid_transform_dict, open("improper_rigid_transformations.pickle", "wb"))
    pickle.dump( proper_rigid_transform_dict, open("proper_rigid_transformations.pickle", "wb"))

except Exception:
    print(space_group)
    raise
