"""
This script is used to query the Bilbao Crystallographic Server for the
chirality-preserving Euclidean normalizers for each space group. This
information is needed so that one common visualization can be decided for a
certain material.

Note that there can be connections problems with the server and it is
recommeneded to do the scraping with some delays and by storing intermediate
results.
"""
import re
import pickle
import numpy as np
import urllib.request
import time
from bs4 import BeautifulSoup


def generate_coset_variants(coset):
    matches = list(re.finditer('[-+]?\d+(?:\/\d+)?', coset))
    start = 0
    variants = [coset]
    if matches:
        variant = []
        for i_match, match in enumerate(matches):
            string = match.group()
            nominator, denominator = string.split('/')
            assert nominator.startswith('+') or nominator.startswith('-')
            nominator = int(nominator)
            denominator = int(denominator)
            new_nominator = nominator - denominator
            new_nominator = "+" if new_nominator >= 0  else '' + str(new_nominator)
            variant.append(coset[start:match.start()])
            variant.append(f'{new_nominator}/{denominator}')
            start = match.end()
            if i_match == len(matches) - 1:
                variant.append(coset[start:])
        variants.append("".join(variant))
    return variants

try:
    # Fetch data for each space group
    start = 201
    end = 230
    groups = range(start, end+1)
    normalizers = {}
    for space_group in groups:
        print(space_group)

        # Space groups 229 and 230 do not have any normalizers.
        if space_group == 229 or space_group == 230:
            continue

        # Read the page containing all the normalizers.
        time.sleep(0.1)
        url = "http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-normsets?from=wycksets&gnum={}".format(space_group)
        html_raw = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(html_raw, 'html.parser')
        tables = soup.body.center.find_all("table")

        # If the page states that the Euclidean ones coincide with affine ones,
        # then simply parse the full table.
        n_cosets = None
        coincides = soup.body.center.findAll(string=re.compile('The affine normalizer coincides with the'))
        if coincides:
            cosets = None
        # Else parse the cosets: prefer chirality-preserving Euclidean normalizers if
        # they are defined, otherwise choose the regular Euclidean ones.
        else:
            url = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-norm?from=norm&gnum={}&norgens=en".format(space_group)
            html_raw = urllib.request.urlopen(url).read()
            soup = BeautifulSoup(html_raw, 'lxml')
            links = soup.body.center.find_all("a", recursive=True)
            href = None
            for link in links:
                text = link.text
                if text.startswith('The cosets representatives of the Euclidean normalizer'):
                    href = link['href']
                elif text.startswith('The cosets representatives of the chirality-preserving Euclidean'):
                    href = link['href']
            cosets = set(href.split('representatives=')[1].split('@&')[0].split('@'))
            n_cosets = len(cosets) - 1 # Identity is not taken into account

        table = tables[1]
        all_rows = table.find_all("tr", recursive=False)
        rows = all_rows[2:]

        # Figure out which column in the html page has the wyckoff letters
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
            coset = tds[1].text

            # We must also generate different variants of the cosets, as they
            # may be reported with a shift of +/-1.
            coset_variants = generate_coset_variants(coset)

            if cosets is not None and not cosets.intersection(coset_variants):
                continue
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

            # Invert the matrix as the Bilbao Crystalographics server stores the
            # inverse matrix as seen from how they transform the Wyckoff letters
            matrix_array = np.array(matrix_array)
            matrix_array = np.linalg.inv(matrix_array)

            # Get the permutation pairs
            wyckoffs = tds[wyckoff_column_number].string
            transformed_letters = wyckoffs.split()
            permutations = {}
            for i, letter in enumerate(transformed_letters):
                original_letter = original_wykoffs[i]
                permutations[original_letter] = letter

            if space_group not in normalizers:
                normalizers[space_group] = []

            normalizers[space_group].append({"transformation": matrix_array, "permutations": permutations})

        # Check that all cosets are found
        if cosets is not None and len(normalizers[space_group]) != n_cosets:
            raise Exception('Could not find all cosets!')

    # Create a pickle file of the translations for later use
    pickle.dump(normalizers, open(f"chirality_preserving_euclidean_normalizers_{start}_{end}.pickle", "wb"))
except Exception:
    print(space_group)
    raise
