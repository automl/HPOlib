#!/usr/bin/env python

##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from argparse import ArgumentParser
from collections import OrderedDict
from StringIO import StringIO
import numpy as np
import sys

try:
    import jinja2
    from jinja2 import Template
except ImportError:
    jinja2 = ""

from HPOlib.Plotting import plot_util
from HPOlib import wrapping_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


template_string = \
"""\\documentclass[landscape]{article} % For LaTeX2

\\usepackage[landscape]{geometry}
\\usepackage{multirow}           % import command \multicolmun
\\usepackage{tabularx}           % Convenient table formatting
\\usepackage{booktabs}           % provides \\toprule, \midrule and \\bottomrule

\\begin{document}

\\begin{table}[t]
\\begin{tabularx}{\\textwidth}{lr{%- for name in result_values -%}|Xr{%- endfor -%}}
\\toprule
\multicolumn{2}{l}{}
{%- for name in result_values -%}
&\multicolumn{2}{c}{\\bf {{name}}}
{%- endfor -%}
\\\\
\\multicolumn{1}{l}{\\bf Experiment} &\multicolumn{1}{r}{\\#evals}
{%- for name in result_values -%}
&\\multicolumn{1}{l}{Valid.\\ loss} &\\multicolumn{1}{r}{Best loss}
{%- endfor -%}
\\\\
\\toprule
{{ experiment }} & {{ evals }}
{%- for name in result_values -%}
{%- set results = result_values[name] -%}
{{ ' & ' }}{% if results['mean_best'] == True %}\\textbf{ {%- endif %}{{results['mean']|round(3, 'floor') }}{% if results['mean_best'] == True %}}{% endif %}$\\pm${{ results['std']|round(3, 'floor')}} & {{results['min']|round(3, 'floor') }}{%- endfor %} \\\\
\\bottomrule
\\end{tabularx}
\\end{table}
\\end{document}
"""

def main(pkl_list, name_list, save="", cut=sys.maxint,
         template_string=template_string, experiment_name="Name",
         num_evals="\\#eval"):
    pickles = plot_util.load_pickles(name_list, pkl_list)
    best_dict, idx_dict, keys = plot_util.get_best_dict(name_list, pickles, cut)
    return generate_tex_template(best_dict, name_list,
                          template_string=template_string, save=save,
                          num_evals=num_evals, experiment_name=experiment_name)


def generate_tex_template(best_dict, name_list, save="",
         template_string=template_string, experiment_name="Name",
         num_evals="\\#eval"):
    tex = StringIO()
    result_values = OrderedDict([(name[0], dict()) for name in name_list])

    means = [np.mean(best_dict[name]) for name in result_values]
    stds = [np.std(best_dict[name]) for name in result_values]
    mins = [np.min(best_dict[name]) for name in result_values]
    maxs = [np.max(best_dict[name]) for name in result_values]

    for name in result_values:
        values = result_values[name]

        values["mean"] = np.mean(best_dict[name])
        values["mean_best"] = True if \
            wrapping_util.float_eq(values["mean"], min(means)) else False

        values["std"] = np.std(best_dict[name])
        values["std_best"] = True if \
            wrapping_util.float_eq(values["std"], min(stds)) else False

        values["min"] = np.min(best_dict[name])
        values["min_best"] = True if\
            wrapping_util.float_eq(values["min"], min(mins)) else False

        values["max"] = np.max(best_dict[name])
        values["max_best"] = True if\
            wrapping_util.float_eq(values["max"], min(maxs)) else False

    if jinja2:
        template = Template(template_string)
        tex.write(template.render(result_values=result_values,
                                  experiment=experiment_name, evals=num_evals))
    else:
        tex.write("Name & #evals")
        for name in result_values:
            values = result_values[name]
            tex.write(" & ")
            tex.write(values["mean"])
            tex.write("$\\pm$")
            tex.write(values["std"])
            tex.write(" & ")
            tex.write(values["min"])
        tex.write("\\\\")

    tex.seek(0)
    table = tex.getvalue()

    if save != "":
        with open(save, "w") as fh:
            fh.write(table)
    else:
        return table


if __name__ == "__main__":
    prog = "python statistics.py WhatIsThis <manyPickles> WhatIsThis <manyPickles> [WhatIsThis <manyPickles>]"
    description = "Generate a LaTeX results table."

    parser = ArgumentParser(description=description, prog=prog)

    parser.add_argument("-c", "--cut", dest="cut", default=sys.maxint,
                        type=int, help="Only consider that many evaluations")
    parser.add_argument("-s", "--save", dest="save", default="",
                        help="Where to save plot instead of showing it?")
    args, unknown = parser.parse_known_args()
    # TODO-list:
    # 1. Add statistical relevance
    # 2. Add parameters to control whether a value should be printed bold
    # face, underlined or whatever
    # 3. Add a custom rounding
    # 4. Determine the experiment name and number of evaluations

    pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(unknown)
    main(pkl_list_main, name_list_main, args.save, args.cut)


