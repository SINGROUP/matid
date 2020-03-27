"""
For each space group fetches information about the Wyckoff positions that have
free parameters, i.e. can be moved freely in some direction. Returns the free
parameters and the algebraic expressions for the locations of the atoms based
on the free variables.
"""
import re
import numpy as np
import pickle
from fractions import Fraction
import urllib.request
from bs4 import BeautifulSoup


regex_multiplication = re.compile("(\d)([xyz])")
regex_expression = re.compile("\(([xyz\d\/\+\- ]+).?,([xyz\d\/\+\- ]+).?,([xyz\d\/\+\- ]+).?\)")
regex_translation = re.compile("\(([\d\/]+),([\d\/]+),([\d\/]+)\)")
data = {}
try:
    # Fetch data for each space group
    n_groups = 230
    groups = range(1, n_groups+1)
    # groups = [160]
    for space_group in groups:

        print("SPACE GROUP: {}".format(space_group))

        # Do a http get request for the data
        url = "http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-wp-list?gnum={}".format(space_group)
        html_raw = urllib.request.urlopen(url).read()

        # Create the soup :)
        soup = BeautifulSoup(html_raw, 'lxml')  # Here we use lxml, because the default html.parser is not working properly...
        center = soup.body.center
        tables = center.find_all("table", recursive=False)

        table = tables[0]
        all_rows = table.find_all("tr", recursive=False)
        translation_text = all_rows[1].text
        n_trans = translation_text.count("+")
        if n_trans >= 1:
            if not translation_text.startswith("(0,0,0)"):
                raise ValueError("The first translation is not 0,0,0, but {}".format(translation_text))
        n_trans = max(translation_text.count("+"), 1)

        # Get the fixed translations
        n_matches = 0
        translation_matches = re.finditer(regex_translation, translation_text)
        translations = []
        for translation_match in translation_matches:
            translation = []
            for coord in translation_match.groups():
                translation.append(float(Fraction(coord)))
            if translation_match.groups() != ("0", "0", "0"):
                translations.append(translation)
                n_matches += 1
        if n_matches == 0:
            translations = None
            n_matches = 1
        else:
            translations = np.array(translations)
            n_matches += 1
        if n_matches != n_trans:
            raise ValueError("Not all the translations were parsed.")

        data[space_group] = {}
        for row in all_rows[2:]:
            tds = row.find_all("td", recursive=False)
            letter = tds[1].text
            multiplicity = int(tds[0].text)
            coordinate = tds[3].text

            variables = set()
            if "x" in coordinate:
                variables.add("x")
            if "y" in coordinate:
                variables.add("y")
            if "z" in coordinate:
                variables.add("z")
            results = re.finditer(regex_expression, coordinate)

            expressions = []
            n_results = 0
            for result in results:
                components = [x.strip() for x in result.groups()]
                for i_coord, coord in enumerate(components):
                    match = regex_multiplication.match(coord)
                    if match:
                        groups = match.groups()
                        number = groups[0]
                        variable = groups[1]
                        components[i_coord] = "{}*{}".format(number, variable)
                expressions.append(components)
                n_results += 1
            if multiplicity != n_trans*n_results:
                print(multiplicity)
                print(n_trans)
                print(n_results)
                raise ValueError("The number of found site does not match the multiplicity.")

            data[space_group][letter] = {
                "variables": variables,
                "expressions": expressions
            }
            data[space_group]["translations"] = translations

except Exception:
    print(space_group)
    raise

# print(data)
with open("free_wyckoff_positions.pickle", "wb") as fout:
    pickle.dump(data, fout)
