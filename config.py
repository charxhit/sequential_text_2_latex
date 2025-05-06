from pathlib import Path

dic = {
    "-": r"-",
    "(": r"(",
    ")": r")",
    "+": r"+",
    "=": r"=",
    "0": r"0",
    "1": r"1",
    "2": r"2",
    "3": r"3",
    "4": r"4",
    "5": r"5",
    "6": r"6",
    "7": r"7",
    "8": r"8",
    "9": r"9",
    "geq": r"\geq",
    "gt": r">",
    "i": r"i",
    "in": r"\in",
    "int": r"\int",
    "j": r"j",
    "leq": r"\le",
    "lt": r"<",
    "neq": r"\neq",
    "pi": r"\Pi",
    "sum": r"\sum",
    "theta": r"\theta",
    "times": r"\times",
    "w": r"w",
    "X": r"\X",
    "y": r"y",
    "z": r"z"
}
dataset = Path.cwd() / 'dataset'
pickle_loc = Path.cwd() / 'pickle'
