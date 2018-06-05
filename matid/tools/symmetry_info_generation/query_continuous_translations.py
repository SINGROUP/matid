"""
For each spacegroup returns the lattice directions in which all the atoms can
be moved without breaking the symmetry.
"""
import re
import json
import urllib.request
from bs4 import BeautifulSoup


regex = re.compile("([xyzrst\+,]+)")
try:
    # Fetch data for each space group
    n_groups = 230
    groups = range(1, n_groups+1)
    continuous_translation_dict = {}
    for space_group in groups:

        # Do a http get request for the data
        url = "http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-normsets?from=wycksets&gnum={}".format(space_group)
        html_raw = urllib.request.urlopen(url).read()

        # Create the soup :)
        soup = BeautifulSoup(html_raw, 'html.parser')
        center = soup.body.center
        tables = center.find_all("table", align="center")

        if len(tables) <= 2:
            continue
        table = tables[2]
        all_rows = table.find_all("tr", recursive=False)

        # See if there are continuous translations
        header = all_rows[0]
        titles = header("th")
        found = False
        for i, title in enumerate(titles):
            title_text = title.text
            if title_text == "Continuous Translations":
                found = True
                break

        if found:
            cont_trans = {"a": False, "b": False, "c": False}
            rows = all_rows[1:]
            for row in rows:
                tds = row.find_all("td", recursive=False)
                interpretation = tds[1]
                content = interpretation.text
                match = regex.match(content)
                if match:
                    coords = match.groups()[0]
                    components = coords.split(",")
                    if len(components[0]) != 1:
                        cont_trans["a"] = True
                    if len(components[1]) != 1:
                        cont_trans["b"] = True
                    if len(components[2]) != 1:
                        cont_trans["c"] = True

            continuous_translation_dict[space_group] = cont_trans

            print("Space group {} has continuous translations: {}".format(space_group, cont_trans))

    # Write the results as a pickle file
    with open("translations_continuous.json", "w") as fout:
        fout.write(json.dump(continuous_translation_dict))

except Exception:
    print(space_group)
    raise
