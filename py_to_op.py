import re
import hashlib
import numpy as np

np.set_printoptions(suppress=True)


def term_label(I, J, pqs, idxs):
    term_str = f"S{I+1}{J+1}"
    if pqs == ():
        return term_str + "__const"
    term_str += "_"
    for op, idx in zip(pqs, idxs):
        term_str += f"_{op.lower()}{idx+2}"
    return term_str


def compress_repeated_segments(label):
    """
    Compress repeated consecutive segments in a string.
    Example: 'q10q10q10' -> '3q10'
    """
    compressed = ""
    i = 0
    length = len(label)

    while i < length:
        # Match a part like "q10" (letters + digits)
        match = re.match(r"([a-zA-Z]+)(\d+)", label[i:])

        if match:
            prefix, number = match.groups()
            segment = prefix + number
            count = 1
            i += len(segment)

            # Check for consecutive repetitions
            while label[i : i + len(segment)] == segment:
                count += 1
                i += len(segment)

            # Append compressed part
            if count > 1:
                compressed += f"{count}{segment}"
            else:
                compressed += segment
        else:
            # If no match, just add the current character
            compressed += label[i]
            i += 1

    return compressed


def short_label(label):
    shortened = label.replace("_", "")
    # shortened = compress_repeated_segments(shortened)

    # hash code it
    shortened = "c" + hashlib.md5(label.encode()).hexdigest()[:15]  # First 8 chars
    return shortened


def process_algebra(vib_dict):
    """
    Function for somplifying the representation of monomials.
    For instance q1*q1 will be represented as q1^2.

    It takes into account commutation relations, so q1*p1*q1 can
    not be simplified to q1^2*p1 or p1*q1^2, its left as-is.
    """
    vib_dict_simplified = {idx: "*".join(term) for idx, term in vib_dict.items()}
    return vib_dict_simplified


def process_monomial(vib_part):
    vib_dict = {}
    prefactor = 1
    for e in vib_part:
        match = re.match(r"([a-zA-Z]+)(\d+)", e)
        operator, idx = [match.group(1), match.group(2)]
        if operator == "pp":
            # operator_str = 'KE' #- 1 / 2 * var('dq')**2 #'KE'
            operator_str = "dq^2"
            prefactor *= -1
        elif operator == "q":
            operator_str = "q"

        elif operator == "p":
            operator_str = "dq"
            prefactor *= -1j
        else:
            raise ValueError(f"Operator type {operator} not recognized.")
        mode_dof = int(idx)
        if mode_dof not in vib_dict:
            vib_dict[mode_dof] = [operator_str]
        else:
            vib_dict[mode_dof].append(operator_str)
    vib_dict = dict(sorted(vib_dict.items()))
    vib_dict_simplified = process_algebra(vib_dict)
    return vib_dict_simplified, prefactor


def mctdh_monomial(vib_part):
    vib_dict, prefactor = process_monomial(vib_part)
    monomial_str = ""
    for dof in vib_dict:
        operator_str = str(vib_dict[dof]).replace("**", "^")
        monomial_str += f"|{dof} {operator_str}"
    return monomial_str, prefactor


def map_terms(block_operator, thresh=1e-12):
    n = block_operator.states
    m = block_operator.modes
    param_dict = {}

    """
    Deprecated code block (using old VibronicHamiltonian object)
    """

    # #loop over the blocks
    # for I in range(n):
    #     for J in range(I, n):
    #         block_terms = block_operator.block(I,J).terms
    #         print(block_terms)
    #         exit()
    #         #loop over the terms in vibrational polynomial for block IJ
    #         for key in block:
    #             coeffs = block[key]
    #             for index in np.ndindex(coeffs.shape):
    #                 if abs(coeffs[index]) >= thresh:
    #                     param_dict[term_label(I,J, key, index)] = coeffs[index]

    """
    Updated code:
    """

    for I in range(int(n)):
        for J in range(int(n)):
            for term in block_operator.block(I, J).ops:

                #shape = (m,) * len(term.ops)
                indices = term.coeffs.nonzero()
                # print(f'number of possible indices: {m**len(term.ops)}')
                # print(f'number of nonzero indices: {len(list(indices))}')
                for index in indices:
                    coeff = term.coeffs[index]
                    if abs(coeff) >= thresh:
                        label = term_label(I, J, term.ops, index)
                        assert label not in param_dict
                        param_dict[term_label(I, J, term.ops, index)] = coeff

    return param_dict


def generate_parameter_section(param_dict, units="ev"):

    input_str = ""
    param_dict_shortened = {}
    for label in param_dict:
        if np.imag(param_dict[label]) != 0:
            print(f"LABEL : {label} HAS IMAGINARY PARAMETER!!: {param_dict[label]}")
        val = np.format_float_positional(
            param_dict[label], precision=20, unique=False, fractional=True, trim="k"
        )
        shortened_label = short_label(label)
        if shortened_label in param_dict_shortened:  # this label has been used before!
            print(
                f"WARNING : Short label {shortened_label} (longform={label}) has already been used. Adding a zero."
            )
            shortened_label += "0"
            if shortened_label in param_dict_shortened:
                raise ValueError(
                    f"Short label {shortened_label} (longform={label}) has already been used. Please revise the labelling."
                )

            # raise ValueError(f'Short label {shortened_label} (longform={label}) has already been used. Please revise the labelling.')
        assert shortened_label not in param_dict_shortened
        param_dict_shortened[shortened_label] = val
        input_str += f"{short_label(label)} = {val} , {units}\n"

    return input_str


def generate_hamiltonian_section(param_dict, m):
    print("NUMBER OF TERMS IN PARAM_DICT")
    print(len(param_dict))
    for key in param_dict:
            print(f'{key} : {param_dict[key]}')
    input_str = "modes | el |"
    for idx in range(m):
        input_str += f" mode{idx+1} "
        if idx < m - 1:
            input_str += "|"

    term_str = "\n"
    for label in param_dict:
        prefactor = 1
        elec, vib = label.split("__")
        I, J = list(elec.replace("S", ""))
        vib_ops = vib.split("_")
        if I != J:
            symmetry = "Z"  # asymmetric element
        else:
            symmetry = "S"  # symmetric element, needed for diagonal
        term_str += f"{short_label(label)} |1 {symmetry}{I}&{J} "
        if vib_ops != ["const"]:
            monomial, prefactor = mctdh_monomial(vib_ops)
            term_str += monomial
        param_dict[label] = param_dict[label] * prefactor
        term_str += "\n"

    input_str += term_str
    return input_str


def generate_op(vibronic_block_operator=None, m=None, run_name=None, param_dict =None):

    if param_dict is None:
        param_dict = map_terms(vibronic_block_operator)

    hamiltonian_section_str = generate_hamiltonian_section(param_dict, m)
    parameter_section_str = generate_parameter_section(param_dict, units="ev")

    maxkoe = 2 * len(param_dict)  # Number of Hamiltonian terms
    maxhtm = int(2 * maxkoe)  # Maximal number of operators for building the Hamiltonian(s)
    maxhop = int(2 * maxkoe)  # Maximal number of labels used in operators

    metadata = {"maxkoe": maxkoe, "maxhtm": maxhtm, "maxhop": maxhop, "maxfac": 600}
    for key in metadata:
        print(f"{key} : {metadata[key]}")

    op_str = f"""
OP_DEFINE-SECTION
title
{run_name}
end-title
end-op_define-section

PARAMETER-SECTION
{parameter_section_str}

end-parameter-section

-----------------------------------------
HAMILTONIAN-SECTION
-----------------------------------------
{hamiltonian_section_str}

end-hamiltonian-section

end-operator

"""

    return op_str, metadata
