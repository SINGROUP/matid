"""
For each space group determines the settings used by spglib and queries the
Wyckoff positions corresponding to them from the Bilbao Crystallographic
Server.

By default spglib will use the lowest Hall number as the setting (origin,
cetring, etc.) because this choice is more unique than the "default" setting.
Because of this choice the corresponding Wyckoff sets differ from the default
settings that are provided for each space group. Luckily, the Bilbao
Crystallographic Server also supports querying the Wyckoff sets in the
ITA-settings, which contains also the settings for each Hall number. This
script thus determines the settings used by spglib (by looking at Seto's
webpage where spglib also gets them.) and then maps the settings to a
correponding ITA setting in Bilbao to fetch the Wyckoff positions.

The mathematical expression for the Wyckoff positions is stored along with the
necessary matrices for solving the variables from a particular system.
"""
import re
import parser
import numpy as np
import pickle
from fractions import Fraction
import urllib.request
from bs4 import BeautifulSoup
import requests


def convert(var):
    val = None
    if var == "-":
        val = -1
    elif var == "+":
        val = 1
    elif var == "":
        val = 1
    elif var is not None:
        code = parser.expr(var).compile()
        val = eval(code)
    return val


def matrixfy(expression, variables):
    # Convert into matrix form
    A = np.zeros((3, 3))
    T = np.zeros((3))
    for i in range(3):
        match = reg.match(expression[i])
        groups = match.groupdict()
        if expression[i] != "0" and len(match.groups()) == 0:
            raise ValueError("Could not find the variable information in nonzero Wyckoff position expression.")
        for j, var in enumerate(["x", "y", "z"]):
            multiplier = convert(groups[var])
            if multiplier is not None:
                A[i,j] = multiplier
            constant = convert(groups["c"])
            if constant is not None:
                T[i] = constant
    A = A.T  # Transposed for doing calculations with row vectors
    return A, T


def extract_url(url, values=None):
    """Given an url, extracts the contents as a BeautifulSoup object.

    Args:
       url: URL to parse.

    Returns:
        The site contents as a BeautifulSoup object
    """
    if values:
        r = requests.post(url, data=values)
    else:
        r = requests.get(url)

    # Create the soup :)
    html_raw = r.text
    soup = BeautifulSoup(html_raw, 'lxml')  # Here we use lxml, because the default html.parser is not working properly...

    return soup


def extract_wyckoff_sets(soup, settings):
    """Given a BeautifulSoup object for a site containing Wyckoff positions,
    extracts them and return them as a python dictionary.

    Args:
        soup: BeautifulSoup object.

    Returns:
        dict: The Wyckoff position information.
    """
    # For ITA settings the page is a bit different
    if settings == "ITA":
        center = soup.body.center
        tables = center.find_all("table", recursive=False)
        table = tables[2].center.table
        column_idx = 4
        row_idx = 3
        row_trans = 2
        col_trans = 1
    elif settings == "default":
        center = soup.body.center
        tables = center.find_all("table", recursive=False)
        table = tables[0]
        column_idx = 3
        row_idx = 2
        row_trans = 1
        col_trans = 0

    data = {}
    all_rows = table.find_all("tr", recursive=False)

    # Get global translations
    translation_columns = all_rows[row_trans].find_all("td", recursive=False)
    if translation_columns[0].text == "":
        translation_text = ""
        n_trans = 0
    else:
        translation_text = translation_columns[col_trans].text
        n_trans = translation_text.count("+")
        if n_trans >= 1:
            if not translation_text.startswith("(0,0,0)"):
                raise ValueError("The first translation is not 0,0,0, but {}".format(translation_text))
    n_trans = max(translation_text.count("+"), 1)

    # Get the fixed translations
    n_matches = 1
    translation_matches = re.finditer(regex_translation, translation_text)
    translations = []
    for translation_match in translation_matches:
        translation = []
        for coord in translation_match.groups():
            translation.append(float(Fraction(coord)))
        if translation_match.groups() != ("0", "0", "0"):
            translations.append(translation)
            n_matches += 1
    translations = np.array(translations)
    if n_matches != n_trans:
        raise ValueError("Not all the translations were parsed.")

    for row in all_rows[row_idx:]:
        tds = row.find_all("td", recursive=False)
        letter = tds[1].text
        multiplicity = int(tds[0].text)
        coordinate = tds[column_idx].text

        variables = set()
        if "x" in coordinate:
            variables.add("x")
        if "y" in coordinate:
            variables.add("y")
        if "z" in coordinate:
            variables.add("z")
        results = re.finditer(regex_expression, coordinate)
        variable_map = {}
        for ivar, variable in enumerate(["x", "y", "z"]):
            if variable in variables:
                variable_map[ivar] = variable

        expressions = []
        matrices = []
        constants = []
        n_results = 0
        for result in results:
            components = [x.strip() for x in result.groups()]
            m, c = matrixfy(components, variables)
            expressions.append(components)
            matrices.append(m)
            constants.append(c)
            n_results += 1

        # Check that the first representative position can be used to
        # determine the variables
        m_repr = matrices[0]
        n_solved = 0
        for idx, var in variable_map.items():
            for icomp in range(3):
                if m_repr[idx][icomp] == 1:
                    n_solved += 1
                    break
        if n_solved != len(variables):
            print(n_solved)
            raise ValueError("The representative position for letter {} cannot be used to determine variables.".format(letter))

        if multiplicity != n_trans*n_results:
            print(multiplicity)
            print(n_trans)
            print(n_results)
            raise ValueError("The number of found site does not match the multiplicity.")

        data[letter] = {
            "variables": variables,
            "expressions": expressions,
            "matrices": np.array(matrices),
            "constants": np.array(constants),
        }
        data["translations"] = translations

    return data

if __name__ == "__main__":
    # Regexes
    regex_multiplication = re.compile("(\d)([xyz])")
    regex_expression = re.compile("\(([xyz\d\/\+\- ]+).?,([xyz\d\/\+\- ]+).?,([xyz\d\/\+\- ]+).?\)")
    regex_translation = re.compile("\(([\d\/]+),([\d\/]+),([\d\/]+)\)")
    regex_bilbao = re.compile("(?P<hm>[^\^\:[\s](?:\s[^\^\:[\s]+)*)(?:\s+\[origin (?P<origin>\d+)\])?(?:\s+\:(?P<centring>\S))?")
    reg = re.compile("(?:(?P<x>[+-]?(?:\d(?:\/\d+)?)?)x)?(?:(?P<y>[+-]?(?:\d(?:\/\d+)?)?)y)?(?:(?P<z>[+-]?(?:\d(?:\/\d+)?)?)z)?(?P<c>[+-]?\d\/?\d*)?")

    # Fetch the setting used by spglib (for each space group the first Hall
    # number setting is used)
    url = "http://pmsl.planet.sci.kobe-u.ac.jp/~seto/?page_id=37&lang=en"
    seto_soup = extract_url(url)

    table = seto_soup.find('tbody')
    all_rows = table.find_all("tr", recursive=False)
    spglib_defaults = {}
    i_spg = 1
    for row in all_rows:
        tds = row.find_all("td", recursive=False)
        try:
            hall_number = int(tds[0].text)
        except Exception:
            continue
        space_group = int(tds[1].text)
        sub_hall = int(tds[2].text)
        settings = tds[3].text

        hms = tds[6].text.split("=")
        if len(hms) == 1:
            hm = hms[0]
        else:
            hm = hms[1]
        hm = hm.strip()

        # There is a small difference on Seto's webpage for space groups 200-206: A minus sign is missing
        if space_group >= 200 and space_group <= 206:
            hm = hm.replace("3", "-3")

        full_notation = tds[7].text

        # Raise exception if match not found for a space group number
        if space_group > i_spg:
            raise

        if space_group == i_spg:
            spglib_defaults[space_group] = [hall_number, sub_hall, hm, settings]
            i_spg += 1

    # Fetch the default settings from Bilbao
    primary_url = "https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-wp-list"
    default_settings = {}
    wyckoff_sets = {}
    for space_group in range(1, 231):
        print(space_group)
        # Do a HTTP POST request for the data
        values = {
            'gnum': space_group,
            'settings': "ITA Settings",
        }
        soup = extract_url(primary_url, values)
        table = soup.find('form').table
        all_rows = table.find_all("tr", recursive=False)

        # Get the correct row
        for row in all_rows[1:]:
            tds = row.find_all("td", recursive=False)
            transform = tds[2].b.text

            # Match with spglib
            setting = tds[1].a.text
            matches = regex_bilbao.match(setting)
            groups = matches.groupdict()
            hm_bilbao = groups.get("hm")
            origin_bilbao = groups.get("origin")
            centring_bilbao = groups.get("centring")
            hall, sub_hall, hm_spglib, settings_spglib = spglib_defaults[space_group]

            # Bilbao uses the letter e to indicate any of the remaining axes in
            # cases where the actual axis letter does not matter. spblib, however,
            # provides the actual axis letter always. Thus we simply match with the
            # first match where e is any valid letter.
            match_regex = re.compile(hm_bilbao.replace("e", "\S"))
            match = False
            if match_regex.match(hm_spglib) is not None:
                print("HM matched: {}={}".format(hm_spglib, hm_bilbao))
                match = True
                if origin_bilbao is not None:
                    match = False
                    origin_bilbao = int(origin_bilbao)
                    origin_spglib = int(settings_spglib[0])
                    if origin_bilbao == origin_spglib:
                        print("Origin matched: {}={}".format(origin_spglib, origin_bilbao))
                        match = True
                if centring_bilbao is not None:
                    match = False
                    centring_spglib = (settings_spglib[0]).lower()
                    if centring_bilbao == centring_spglib:
                        print("Centring matched: {}={}".format(centring_spglib, centring_bilbao))
                        match = True

            # If match found, follow the link to extract the Wyckoff positions
            if match:
                print("Matched!")
                link = tds[1].a["href"]

                # If link points to ITA structure it will work fine. If it however
                # points to the default structure we have to modify it a bit.
                if "trgen" in link:
                    values = None
                    settings = "ITA"
                else:
                    link = "/cgi-bin/cryst/programs/nph-wp-list"
                    values = {
                        'gnum': space_group,
                        'standard': "Standard/Default Setting",
                    }
                    settings = "default"
                url = "https://www.cryst.ehu.es" + link
                bs = extract_url(url, values)
                wyckoff_set = extract_wyckoff_sets(bs, settings)
                wyckoff_sets[space_group] = wyckoff_set
                break

        if not match:
            raise ValueError("Match not found between spglib and default settings for space group {}.".format(space_group))

    # Save the found Wyckoff sets as a pickle file
    with open("wyckoff_sets.pickle", "wb") as fout:
        pickle.dump(wyckoff_sets, fout)
