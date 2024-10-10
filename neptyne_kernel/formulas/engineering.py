import cmath
from enum import Enum
from functools import reduce, wraps
from math import erf, erfc
from operator import add, and_, lshift, mul, or_, pow, rshift, sub, truediv, xor

from scipy.special import iv, jv, kn, yn

from ..cell_range import CellRange
from ..spreadsheet_error import NA_ERROR, NUM_ERROR, VALUE_ERROR, SpreadsheetError
from .helpers import Numeric, _flatten_range

__all__ = [
    "BESSELI",
    "BESSELJ",
    "BESSELK",
    "BESSELY",
    "BIN2DEC",
    "BIN2HEX",
    "BIN2OCT",
    "BITAND",
    "BITLSHIFT",
    "BITOR",
    "BITRSHIFT",
    "BITXOR",
    "COMPLEX",
    "CONVERT",
    "DEC2BIN",
    "DEC2HEX",
    "DEC2OCT",
    "DELTA",
    "ERF",
    "ERFC",
    "GESTEP",
    "HEX2BIN",
    "HEX2DEC",
    "HEX2OCT",
    "IMABS",
    "IMAGINARY",
    "IMARGUMENT",
    "IMCONJUGATE",
    "IMCOS",
    "IMCOSH",
    "IMCOT",
    "IMCSC",
    "IMCSCH",
    "IMDIV",
    "IMEXP",
    "IMLN",
    "IMLOG10",
    "IMLOG2",
    "IMPOWER",
    "IMPRODUCT",
    "IMREAL",
    "IMSEC",
    "IMSECH",
    "IMSIN",
    "IMSINH",
    "IMSQRT",
    "IMSUB",
    "IMSUM",
    "IMTAN",
    "OCT2BIN",
    "OCT2DEC",
    "OCT2HEX",
]

ComplexNum = Numeric | str

_CHAR_LIMIT = 10
_INV_MATCHENV = 1764


class CDC(Enum):
    Mass = 0
    Length = 1
    Time = 2
    Pressure = 3
    Force = 4
    Energy = 5
    Power = 6
    Magnetism = 7
    Temperature = 8
    Volume = 9
    Area = 10
    Speed = 11
    Information = 12


_DECIMAL_PREFIXES = {
    "y": -24,
    "z": -21,
    "a": -18,
    "f": -15,
    "p": -12,
    "n": -9,
    "u": -6,
    "m": -3,
    "c": -2,
    "d": 1,
    "e": 1,
    "h": 2,
    "k": 3,
    "M": 6,
    "G": 9,
    "T": 12,
    "P": 15,
    "E": 18,
    "Z": 21,
    "Y": 24,
}

_BINARY_PREFIXES = {
    "k": 10,
    "M": 20,
    "G": 30,
    "T": 40,
    "P": 50,
    "E": 60,
    "Z": 70,
    "Y": 80,
}


class ConvertData:
    def __init__(
        self,
        unit_name: str,
        convert_constant: Numeric,
        eclass: CDC,
        pref_support: bool = False,
    ):
        self.const = convert_constant
        self.name = unit_name
        self.eclass = eclass
        self.pref_support = pref_support

    def get_matching_level(self, ref: str):
        astr = ref[:]
        nindex = ref[::-1].find("^")
        if nindex == len(astr) - 2:
            astr = astr.replace("^", "")
        if astr == self.name:
            return 0
        nlen = len(astr)
        bpref = self.pref_support
        bonechar = bpref and nlen > 1 and self.name == astr[1:]
        if bonechar or (
            bpref
            and nlen > 2
            and (self.name == astr[2:])
            and astr[0] == "d"
            and astr[1] == "a"
        ):
            p = astr[0]
            n = _DECIMAL_PREFIXES.get(p, _INV_MATCHENV)

            if bonechar and p == "d":
                n = -n

            if n != _INV_MATCHENV:
                clast = astr[-1]
                if clast == "2":
                    n *= 2
                elif clast == "3":
                    n *= 3
            return n

        elif nlen > 2 and self.name == astr[2:] and self.eclass == CDC.Information:
            if astr[1] != "i":
                return _INV_MATCHENV
            return _BINARY_PREFIXES.get(astr[0], _INV_MATCHENV)

        else:
            return _INV_MATCHENV

    def convert(
        self,
        val: Numeric,
        r_to: "ConvertData",
        match_level_from: int,
        match_level_to: int,
    ):
        if self.eclass != r_to.eclass:
            return NA_ERROR

        bin_from_lev = match_level_from > 0 and (match_level_from % 10) == 0
        bin_to_lev = match_level_to > 0 and (match_level_to % 10) == 0

        if self.eclass == CDC.Information and (bin_from_lev or bin_to_lev):
            if bin_from_lev and bin_to_lev:
                lev_from = match_level_from - match_level_to
                val *= r_to.const / self.const
                if lev_from:
                    val *= 2.0**lev_from
            elif bin_from_lev:
                val *= (r_to.const / self.const) * (
                    2**match_level_from / 10**match_level_to
                )
            else:
                val *= (r_to.const / self.const) * (
                    10**match_level_from / 2**match_level_to
                )
            return val

        match_level_from = match_level_from - match_level_to
        val *= r_to.const / self.const

        if match_level_from:
            val *= 10**match_level_from

        return val

    def convert_from_base(self, val: Numeric, match_level_to: int):
        return val * self.const * 10 ** (-match_level_to)


class ConvertDataLinear(ConvertData):
    def __init__(
        self,
        unit_name: str,
        convert_constant: Numeric,
        convert_offset: Numeric,
        eclass: CDC,
        pref_support=False,
    ):
        super().__init__(unit_name, convert_constant, eclass, pref_support)
        self.offset = convert_offset

    # for cases where f(x) = a + bx applies (e.g. Temperatures)
    def convert(self, val: Numeric, r_to, match_level_from: int, match_level_to: int):
        if self.eclass != r_to.eclass:
            return NA_ERROR
        return r_to.convert_from_base(
            self.convert_to_base(val, match_level_from), match_level_to
        )

    def convert_from_base(self, val: Numeric, match_level_to: int):
        val += self.offset
        val *= self.const

        if match_level_to:
            val *= 10 ** (-match_level_to)

        return val

    def convert_to_base(self, val: Numeric, match_level_to: int):
        if match_level_to:
            val *= val**match_level_to

        val /= self.const
        val -= self.offset

        return val


# MASS: 1 Gram is...
_CONVERT_DATA = (
    ConvertData("g", 1.0, CDC.Mass, True),  # Gram
    ConvertData("sg", 6.8522050005347800e-05, CDC.Mass),  # Pieces
    ConvertData("lbm", 2.2046229146913400e-03, CDC.Mass),  # Pound (commercial weight)
    ConvertData("u", 6.0221370000000000e23, CDC.Mass, True),  # U (atomic mass)
    ConvertData("ozm", 3.5273971800362700e-02, CDC.Mass),  # Ounce (commercial weight)
    ConvertData("stone", 1.574730e-04, CDC.Mass),  # *** Stone
    ConvertData("ton", 1.102311e-06, CDC.Mass),  # *** Ton
    ConvertData("grain", 1.543236e01, CDC.Mass),  # *** Grain
    ConvertData("pweight", 7.054792e-01, CDC.Mass),  # *** Pennyweight
    ConvertData("brton", 9.842065e-07, CDC.Mass),  # *** Gross Registered Ton
    ConvertData("cwt", 2.2046226218487758e-05, CDC.Mass),  # U.S. (short) hundredweight
    ConvertData(
        "shweight", 2.2046226218487758e-05, CDC.Mass
    ),  # U.S. (short) hundredweight also
    ConvertData("uk_cwt", 1.9684130552221213e-05, CDC.Mass),  # Imperial hundredweight
    ConvertData(
        "lcwt", 1.9684130552221213e-05, CDC.Mass
    ),  # Imperial hundredweight also
    ConvertData(
        "hweight", 1.9684130552221213e-05, CDC.Mass
    ),  # Imperial hundredweight also
    ConvertData("uk_ton", 9.8420652761106063e-07, CDC.Mass),  # Imperial ton
    ConvertData("LTON", 9.8420652761106063e-07, CDC.Mass),  # Imperial ton also
    # LENGTH: 1 Meter is...
    ConvertData("m", 1.0, CDC.Length, True),  # Meter
    ConvertData("mi", 6.2137119223733397e-04, CDC.Length),  # Britsh Mile
    ConvertData("Nmi", 5.3995680345572354e-04, CDC.Length),  # Nautical Mile
    ConvertData("in", 3.9370078740157480e01, CDC.Length),  # Inch
    ConvertData("ft", 3.2808398950131234e00, CDC.Length),  # Foot
    ConvertData("yd", 1.0936132983377078e00, CDC.Length),  # Yard
    ConvertData("ang", 1.0e10, CDC.Length, True),  # Angstrom
    ConvertData("Pica", 2.8346456692913386e03, CDC.Length),  # Pica Point (1/72 Inch)
    ConvertData("picapt", 2.8346456692913386e03, CDC.Length),  # Pica Point (1/72 Inch)
    ConvertData("pica", 2.36220472441e02, CDC.Length),  # pica (1/6 Inch)
    ConvertData("ell", 8.748906e-01, CDC.Length),  # *** Ell
    ConvertData("parsec", 3.240779e-17, CDC.Length, True),  # *** Parsec
    ConvertData("pc", 3.240779e-17, CDC.Length, True),  # *** Parsec also
    ConvertData(
        "lightyear", 1.0570234557732930e-16, CDC.Length, True
    ),  # *** Light Year
    ConvertData("ly", 1.0570234557732930e-16, CDC.Length, True),  # *** Light Year also
    ConvertData("survey_mi", 6.2136994949494949e-04, CDC.Length),  # U.S. survey mile
    # TIME: 1 Second is...
    ConvertData("yr", 3.1688087814028950e-08, CDC.Time),  # Year
    ConvertData("day", 1.1574074074074074e-05, CDC.Time),  # Day
    ConvertData("d", 1.1574074074074074e-05, CDC.Time),  # Day also
    ConvertData("hr", 2.7777777777777778e-04, CDC.Time),  # Hour
    ConvertData("mn", 1.6666666666666667e-02, CDC.Time),  # Minute
    ConvertData("min", 1.6666666666666667e-02, CDC.Time),  # Minute also
    ConvertData("sec", 1.0, CDC.Time, True),  # Second
    ConvertData("s", 1.0, CDC.Time, True),  # Second also
    # PRESSURE, 1 Pascal is...
    ConvertData("Pa", 1.0, CDC.Pressure, True),  # Pascal
    ConvertData("atm", 9.8692329999819300e-06, CDC.Pressure, True),  # Atmosphere
    ConvertData("at", 9.8692329999819300e-06, CDC.Pressure, True),  # Atmosphere also
    ConvertData("mmHg", 7.5006170799862700e-03, CDC.Pressure, True),  # mm Hg (Mercury)
    ConvertData("Torr", 7.5006380000000000e-03, CDC.Pressure),  # *** Torr
    ConvertData("psi", 1.4503770000000000e-04, CDC.Pressure),  # *** Psi
    # FORCE, 1 Newton is...
    ConvertData("N", 1.0, CDC.Force, True),  # Newton
    ConvertData("dyn", 1.0e05, CDC.Force, True),  # Dyn
    ConvertData("dy", 1.0e05, CDC.Force, True),  # Dyn also
    ConvertData("lbf", 2.24808923655339e-01, CDC.Force),  # Pound-Force
    ConvertData("pond", 1.019716e02, CDC.Force, True),  # *** Pond
    # ENERGY, 1 Joule is...
    ConvertData("J", 1.0, CDC.Energy, True),  # Joule
    ConvertData(
        "e", 1.0e07, CDC.Energy, True
    ),  # Erg  -> https,//en.wikipedia.org/wiki/Erg
    ConvertData(
        "c", 2.3900624947346700e-01, CDC.Energy, True
    ),  # Thermodynamical Calorie
    ConvertData("cal", 2.3884619064201700e-01, CDC.Energy, True),  # Calorie
    ConvertData("eV", 6.2414570000000000e18, CDC.Energy, True),  # Electronvolt
    ConvertData("ev", 6.2414570000000000e18, CDC.Energy, True),  # Electronvolt also
    ConvertData("HPh", 3.7250611111111111e-07, CDC.Energy),  # Horsepower Hours
    ConvertData("hh", 3.7250611111111111e-07, CDC.Energy),  # Horsepower Hours also
    ConvertData("Wh", 2.7777777777777778e-04, CDC.Energy, True),  # Watt Hours
    ConvertData("wh", 2.7777777777777778e-04, CDC.Energy, True),  # Watt Hours also
    ConvertData("flb", 2.37304222192651e01, CDC.Energy),  # Foot Pound
    ConvertData("BTU", 9.4781506734901500e-04, CDC.Energy),  # British Thermal Unit
    ConvertData("btu", 9.4781506734901500e-04, CDC.Energy),  # British Thermal Unit also
    # POWER, 1 Watt is...
    ConvertData("W", 1.0, CDC.Power, True),  # Watt
    ConvertData("w", 1.0, CDC.Power, True),  # Watt also
    ConvertData("HP", 1.341022e-03, CDC.Power),  # Horsepower
    ConvertData("h", 1.341022e-03, CDC.Power),  # Horsepower also
    ConvertData("PS", 1.359622e-03, CDC.Power),  # *** German Pferdestaerke
    # MAGNETISM, 1 Tesla is...
    ConvertData("T", 1.0, CDC.Magnetism, True),  # Tesla
    ConvertData("ga", 1.0e04, CDC.Magnetism, True),  # Gauss
    # TEMPERATURE: 1 Kelvin is...
    ConvertDataLinear("C", 1.0, -2.7315000000000000e02, CDC.Temperature),  # Celsius
    ConvertDataLinear(
        "cel", 1.0, -2.7315000000000000e02, CDC.Temperature
    ),  # Celsius also
    ConvertDataLinear("F", 1.8, -2.5537222222222222e02, CDC.Temperature),  # Fahrenheit
    ConvertDataLinear(
        "fah", 1.8, -2.5537222222222222e02, CDC.Temperature
    ),  # Fahrenheit also
    ConvertDataLinear("K", 1.0, 0.0, CDC.Temperature, True),  # Kelvin
    ConvertDataLinear("kel", 1.0, 0.0, CDC.Temperature, True),  # Kelvin also
    ConvertDataLinear(
        "Reau", 8.0e-01, -2.7315000000000000e02, CDC.Temperature
    ),  # *** Reaumur
    ConvertDataLinear("Rank", 1.8, 0.0, CDC.Temperature),  # *** Rankine
    # VOLUME, 1 Liter is...
    ConvertData("tsp", 2.0288413621105798e02, CDC.Volume),  # US teaspoon 1/768 gallon
    ConvertData("tbs", 6.7628045403685994e01, CDC.Volume),  # US tablespoon 1/256 gallon
    ConvertData("oz", 3.3814022701842997e01, CDC.Volume),  # Ounce Liquid 1/128 gallon
    ConvertData("cup", 4.2267528377303746e00, CDC.Volume),  # Cup 1/16 gallon
    ConvertData("pt", 2.1133764188651873e00, CDC.Volume),  # US Pint 1/8 gallon
    ConvertData("us_pt", 2.1133764188651873e00, CDC.Volume),  # US Pint also
    ConvertData(
        "uk_pt", 1.7597539863927023e00, CDC.Volume
    ),  # UK Pint 1/8 imperial gallon
    ConvertData(
        "qt", 1.0566882094325937e00, CDC.Volume
    ),  # Quart                  1/4 gallon
    ConvertData(
        "gal", 2.6417205235814842e-01, CDC.Volume
    ),  # Gallon                 1/3.785411784
    ConvertData("l", 1.0, CDC.Volume, True),  # Liter
    ConvertData("L", 1.0, CDC.Volume, True),  # Liter also
    ConvertData("lt", 1.0, CDC.Volume, True),  # Liter also
    ConvertData("m3", 1.0e-03, CDC.Volume, True),  # *** Cubic Meter
    ConvertData("mi3", 2.3991275857892772e-13, CDC.Volume),  # *** Cubic Britsh Mile
    ConvertData("Nmi3", 1.5742621468581148e-13, CDC.Volume),  # *** Cubic Nautical Mile
    ConvertData("in3", 6.1023744094732284e01, CDC.Volume),  # *** Cubic Inch
    ConvertData("ft3", 3.5314666721488590e-02, CDC.Volume),  # *** Cubic Foot
    ConvertData("yd3", 1.3079506193143922e-03, CDC.Volume),  # *** Cubic Yard
    ConvertData("ang3", 1.0e27, CDC.Volume, True),  # *** Cubic Angstrom
    ConvertData(
        "Pica3", 2.2776990435870636e07, CDC.Volume
    ),  # *** Cubic Pica Point (1/72 inch)
    ConvertData(
        "picapt3", 2.2776990435870636e07, CDC.Volume
    ),  # *** Cubic Pica Point (1/72 inch)
    ConvertData("pica3", 1.31811287245e04, CDC.Volume),  # *** Cubic Pica (1/6 inch)
    ConvertData("barrel", 6.2898107704321051e-03, CDC.Volume),  # *** Barrel (=42gal)
    ConvertData("bushel", 2.837759e-02, CDC.Volume),  # *** Bushel
    ConvertData("regton", 3.531467e-04, CDC.Volume),  # *** Register ton
    ConvertData("GRT", 3.531467e-04, CDC.Volume),  # *** Register ton also
    ConvertData("Schooner", 2.3529411764705882e00, CDC.Volume),  # *** austr. Schooner
    ConvertData("Middy", 3.5087719298245614e00, CDC.Volume),  # *** austr. Middy
    ConvertData("Glass", 5.0, CDC.Volume),  # *** austr. Glass
    ConvertData("Sixpack", 0.5, CDC.Volume),  # ***
    ConvertData("Humpen", 2.0, CDC.Volume),  # ***
    ConvertData("ly3", 1.1810108125623799e-51, CDC.Volume),  # *** Cubic light-year
    ConvertData("MTON", 1.4125866688595436e00, CDC.Volume),  # *** Measurement ton
    ConvertData("tspm", 2.0e02, CDC.Volume),  # *** Modern teaspoon
    ConvertData(
        "uk_gal", 2.1996924829908779e-01, CDC.Volume
    ),  # U.K. / Imperial gallon 1/4.54609
    ConvertData(
        "uk_qt", 8.7987699319635115e-01, CDC.Volume
    ),  # U.K. / Imperial quart  1/4 imperial gallon
    # 1 Square Meter is...
    ConvertData("m2", 1.0, CDC.Area, True),  # *** Square Meter
    ConvertData("mi2", 3.8610215854244585e-07, CDC.Area),  # *** Square Britsh Mile
    ConvertData("Nmi2", 2.9155334959812286e-07, CDC.Area),  # *** Square Nautical Mile
    ConvertData("in2", 1.5500031000062000e03, CDC.Area),  # *** Square Inch
    ConvertData("ft2", 1.0763910416709722e01, CDC.Area),  # *** Square Foot
    ConvertData("yd2", 1.1959900463010803e00, CDC.Area),  # *** Square Yard
    ConvertData("ang2", 1.0e20, CDC.Area, True),  # *** Square Angstrom
    ConvertData(
        "Pica2", 8.0352160704321409e06, CDC.Area
    ),  # *** Square Pica Point (1/72 inch)
    ConvertData(
        "picapt2", 8.0352160704321409e06, CDC.Area
    ),  # *** Square Pica Point (1/72 inch)
    ConvertData("pica2", 5.58001116002232e04, CDC.Area),  # *** Square Pica (1/6 inch)
    ConvertData("Morgen", 4.0e-04, CDC.Area),  # *** Morgen
    ConvertData("ar", 1.0e-02, CDC.Area, True),  # *** Ar
    ConvertData("acre", 2.471053815e-04, CDC.Area),  # *** Acre
    ConvertData("uk_acre", 2.4710538146716534e-04, CDC.Area),  # *** International acre
    ConvertData(
        "us_acre", 2.4710439304662790e-04, CDC.Area
    ),  # *** U.S. survey/statute acre
    ConvertData("ly2", 1.1172985860549147e-32, CDC.Area),  # *** Square Light-year
    ConvertData("ha", 1.0e-04, CDC.Area),  # *** Hectare
    # SPEED, 1 Meter per Second is...
    ConvertData("m/s", 1.0, CDC.Speed, True),  # *** Meters per Second
    ConvertData("m/sec", 1.0, CDC.Speed, True),  # *** Meters per Second also
    ConvertData("m/h", 3.6e03, CDC.Speed, True),  # *** Meters per Hour
    ConvertData("m/hr", 3.6e03, CDC.Speed, True),  # *** Meters per Hour also
    ConvertData("mph", 2.2369362920544023e00, CDC.Speed),  # *** Britsh Miles per Hour
    ConvertData(
        "kn", 1.9438444924406048e00, CDC.Speed
    ),  # *** Knot = Nautical Miles per Hour
    ConvertData("admkn", 1.9438446603753486e00, CDC.Speed),  # *** Admiralty Knot
    ConvertData("ludicrous speed", 2.0494886343432328e-14, CDC.Speed),
    ConvertData("ridiculous speed", 4.0156958471424288e-06, CDC.Speed),
    # INFORMATION, 1 Bit is...
    ConvertData("bit", 1.0, CDC.Information, True),  # *** Bit
    ConvertData("byte", 1.25e-01, CDC.Information, True),  # *** Byte
)


def str2complex(src: str) -> complex | SpreadsheetError:
    if not isinstance(src, str):
        return complex(src), "i"
    src = src.replace(" ", "")
    if "i" in src:
        src = src.replace("i", "j")
        suffix = "i"
    else:
        suffix = "j"
    try:
        return complex(src), suffix
    except ValueError:
        return NUM_ERROR


def complex2str(src: complex, suffix: str = "i") -> ComplexNum | SpreadsheetError:
    """Convert src to str if src is a complex number"""

    def num_by_type(val):
        return int(val.real) if not val.real % 1 else val.real

    if not isinstance(src, complex):
        return src
    real = num_by_type(src.real)
    imag = num_by_type(src.imag)
    result = ""
    if real:
        result += str(real)
    if imag:
        if real and imag > 0:
            result += "+"
        if imag == -1:
            result += "-"
        elif imag != 1:
            result += str(imag)
        result += suffix
    if not real and not imag:
        return "0"
    return result


def bessel_func(func):
    def decorator(f):
        @wraps(f)
        def wrapper(x, n):
            if n < 0:
                return NUM_ERROR
            return func(int(n), x)

        return wrapper

    return decorator


@bessel_func(iv)
def BESSELI(x: Numeric, n: Numeric) -> Numeric:
    """Returns the modified Bessel function In(x)"""
    pass


@bessel_func(jv)
def BESSELJ(x: Numeric, n: Numeric) -> Numeric:
    """Returns the Bessel function Jn(x)"""
    pass


@bessel_func(lambda x, n: kn(x, n) if x > 0 else NUM_ERROR)
def BESSELK(x: Numeric, n: Numeric) -> Numeric:
    """Returns the modified Bessel function Kn(x)"""
    pass


@bessel_func(lambda x, n: yn(x, n) if x > 0 else NUM_ERROR)
def BESSELY(x: Numeric, n: Numeric) -> Numeric:
    """Returns the Bessel function Yn(x)"""
    pass


def to_decimal(src: str, base: int, char_lim: int = 10) -> int:
    if not 2 <= base <= 36 or len(src) > char_lim:
        return NUM_ERROR
    val = 0
    first_dig = 0
    is_first_dig = True

    for p in src:
        if "0" <= p <= "9":
            n = ord(p) - ord("0")
        elif "A" <= p <= "Z":
            n = 10 + ord(p) - ord("A")
        elif "a" <= p <= "z":
            n = 10 + ord(p) - ord("a")
        else:
            n = base
        if n >= base:
            return NUM_ERROR
        if is_first_dig:
            is_first_dig = False
            first_dig = n
        val = val * base + n

    if len(src) == char_lim and not is_first_dig and first_dig >= base // 2:
        val = -(base**char_lim - val)

    return val


_bases = {2: (bin, "1"), 8: (oct, "7"), 16: (hex, "F")}


def from_decimal(
    num: Numeric, base: int, n_places: int, max_places: int, use_places: bool
):
    num = int(num)
    is_neg = num < 0
    if is_neg:
        num = base**max_places + num
    if base in _bases:
        res = _bases[base][0](num)[2:].upper()

    if use_places:
        nlen = len(res)
        if n_places is None:
            n_places = nlen
        if not is_neg and nlen > n_places:
            return NUM_ERROR
        elif (is_neg and nlen < max_places) or (not is_neg and nlen < n_places):
            n_left = n_places - nlen
            res = (_bases[base][1] if is_neg else "0") * n_left + res
    return res


def convert_base(base_from, base_to):
    def decorator(f):
        @wraps(f)
        def wrapper(number, places):
            if isinstance(places, int) and places < 0:
                return NUM_ERROR
            decimal = to_decimal(number, base_from, _CHAR_LIMIT)
            if isinstance(decimal, SpreadsheetError):
                return decimal
            return from_decimal(decimal, base_to, places, _CHAR_LIMIT, True)

        return wrapper

    return decorator


_MAX_BITWISE = 281474976710655
_MAX_SHIFT = 53


def bitwise(func, check_both=True):
    def decorator(f):
        @wraps(f)
        def wrapper(number1, number2):
            def check_number(number):
                return 0 <= number <= _MAX_BITWISE

            if not check_number(number1):
                return NUM_ERROR
            if check_both and not check_number(number2):
                return NUM_ERROR
            return func(number1, number2)

        return wrapper

    return decorator


def BIN2DEC(number: str) -> int:
    """Converts a binary number to decimal"""
    return to_decimal(number, 2, _CHAR_LIMIT)


@convert_base(2, 16)
def BIN2HEX(number: str, places: int | None = None) -> str:
    """Converts a binary number to hexadecimal"""
    pass


@convert_base(2, 8)
def BIN2OCT(number: str, places: int | None = None) -> str:
    """Converts a binary number to octal"""
    pass


@bitwise(and_)
def BITAND(number1: int, number2: int) -> int:
    """Returns a 'Bitwise And' of two numbers"""
    pass


@bitwise(
    lambda n1, n2: lshift(n1, n2)
    if 0 <= n2 <= _MAX_SHIFT
    else rshift(n1, -n2)
    if n2 < 0
    else NUM_ERROR,
    check_both=False,
)
def BITLSHIFT(number1: int, number2: int) -> int:
    """Returns a value number shifted left by shift_amount bits"""
    pass


@bitwise(or_)
def BITOR(number1: int, number2: int) -> int:
    """Returns a bitwise OR of 2 numbers"""
    pass


@bitwise(
    lambda n1, n2: rshift(n1, n2)
    if 0 <= n2 <= _MAX_SHIFT
    else lshift(n1, -n2)
    if n2 < 0
    else NUM_ERROR,
    check_both=False,
)
def BITRSHIFT(number1: int, number2: int) -> int:
    """Returns a value number shifted right by shift_amount bits"""
    pass


@bitwise(xor)
def BITXOR(number1: int, number2: int) -> int:
    """Returns a bitwise 'Exclusive Or' of two numbers"""
    pass


def COMPLEX(real_num: Numeric, i_num: Numeric, suffix: str = "i") -> ComplexNum:
    """Converts real and imaginary coefficients into a complex number"""
    if suffix not in ["i", "j"]:
        return VALUE_ERROR
    return complex2str(complex(real_num, i_num), suffix)


def CONVERT(number: Numeric, from_unit: str, to_unit: str) -> Numeric:
    """Converts a number from one measurement system to another"""
    p_from = None
    p_to = None
    search_from = True
    search_to = True
    level_from = 0
    level_to = 0

    for item in _CONVERT_DATA:
        if search_from:
            n = item.get_matching_level(from_unit)
            if n != _INV_MATCHENV:
                p_from = item
                level_from = n
                if not n:
                    search_from = False
        if search_to:
            n = item.get_matching_level(to_unit)
            if n != _INV_MATCHENV:
                p_to = item
                level_to = n
                if not n:
                    search_to = False
        if not search_from and not search_to:
            break

    if not p_from or not p_to:
        return NA_ERROR

    return p_from.convert(number, p_to, level_from, level_to)


def DEC2BIN(number: int, places: int | None = None) -> str:
    """Converts a decimal number to binary"""
    if not -512 <= number <= 511:
        return NUM_ERROR
    return from_decimal(number, 2, places, _CHAR_LIMIT, True)


def DEC2HEX(number: int, places: int | None = None) -> str:
    """Converts a decimal number to hexadecimal"""
    if not -549755813888 <= number <= 549755813887:
        return NUM_ERROR
    return from_decimal(number, 16, places, _CHAR_LIMIT, True)


def DEC2OCT(number: int, places: int | None = None) -> str:
    """Converts a decimal number to octal"""
    if not -536870912 <= number <= 536870911:
        return NUM_ERROR
    return from_decimal(number, 8, places, _CHAR_LIMIT, True)


def DELTA(number1: Numeric, number2: Numeric = 0) -> int:
    """Tests whether two values are equal"""
    return int(number1 == number2)


def ERF(lower_limit: Numeric, upper_limit: Numeric | None = None) -> Numeric:
    """Returns the error function"""
    elow = erf(lower_limit)
    return elow if upper_limit is None else erf(upper_limit) - elow


def ERF_PRECISE(x: Numeric) -> Numeric:
    """Returns the error function"""
    return erf(x)


ERF.PRECISE = ERF_PRECISE


def ERFC(x: Numeric) -> Numeric:
    """Returns the complementary error function"""
    return erfc(x)


def ERFC_PRECISE(x: Numeric) -> Numeric:
    """Returns the complementary ERF function integrated between x and infinity"""
    return erfc(x)


ERFC.PRECISE = ERFC_PRECISE


def GESTEP(number: Numeric, step: Numeric = 0) -> int:
    """Tests whether a number is greater than a threshold value"""
    return int(number >= step)


@convert_base(16, 2)
def HEX2BIN(number: str, places: int | None = None) -> str:
    """Converts a hexadecimal number to binary"""
    pass


def HEX2DEC(number: str) -> int:
    """Converts a hexadecimal number to decimal"""
    return to_decimal(number, 16, _CHAR_LIMIT)


@convert_base(16, 8)
def HEX2OCT(number: str, places: int | None = None) -> str:
    """Converts a hexadecimal number to octal"""
    pass


def complex_func(func, result2str=False, complex_args=True, range_args=False):
    def decorator(f):
        @wraps(f)
        def wrapper(*args):
            new_args = []
            suffixes = []
            if range_args:
                for arg in _flatten_range(args):
                    comp = str2complex(arg)
                    if isinstance(comp, SpreadsheetError):
                        return comp
                    new_args.append(comp[0])
                    suffixes.append(comp[1])
            else:
                args_to_convert = (
                    range(len(args)) if complex_args is True else complex_args
                )
                for i, arg in enumerate(args):
                    if i in args_to_convert:
                        comp = str2complex(arg)
                        if isinstance(comp, SpreadsheetError):
                            return comp
                        new_args.append(comp[0])
                        suffixes.append(comp[1])
                    else:
                        new_args.append(arg)
            result = func(*new_args)
            if isinstance(result, SpreadsheetError):
                return result
            return complex2str(result, suffix=suffixes[0]) if result2str else result

        return wrapper

    return decorator


@complex_func(abs)
def IMABS(inumber: ComplexNum) -> Numeric:
    """Returns the absolute value (modulus) of a complex number"""
    pass


@complex_func(lambda x: x.imag)
def IMAGINARY(inumber: ComplexNum) -> Numeric:
    """Returns the imaginary coefficient of a complex number"""
    pass


@complex_func(lambda x: cmath.phase(x))
def IMARGUMENT(inumber: ComplexNum) -> Numeric:
    """Returns the argument theta, an angle expressed in radians"""
    pass


@complex_func(lambda x: x.conjugate(), result2str=True)
def IMCONJUGATE(inumber: ComplexNum) -> ComplexNum:
    """Returns the complex conjugate of a complex number"""
    pass


@complex_func(cmath.cos, result2str=True)
def IMCOS(inumber: ComplexNum) -> ComplexNum:
    """Returns the cosine of a complex number"""
    pass


@complex_func(cmath.cosh, result2str=True)
def IMCOSH(inumber: ComplexNum) -> ComplexNum:
    """Returns the hyperbolic cosine of a complex number"""
    pass


@complex_func(lambda x: 1 / cmath.tan(x) if x != 0 else NUM_ERROR, result2str=True)
def IMCOT(inumber: ComplexNum) -> ComplexNum:
    """Returns the cotangent of a complex number"""
    pass


@complex_func(lambda x: 1 / cmath.sin(x) if x != 0 else NUM_ERROR, result2str=True)
def IMCSC(inumber: ComplexNum) -> ComplexNum:
    """Returns the cosecant of a complex number"""
    pass


@complex_func(lambda x: 1 / cmath.sinh(x) if x != 0 else NUM_ERROR, result2str=True)
def IMCSCH(inumber: ComplexNum) -> ComplexNum:
    """Returns the hyperbolic cosecant of a complex number"""
    pass


@complex_func(lambda x, y: truediv(x, y) if y != 0 else NUM_ERROR, result2str=True)
def IMDIV(inumber1: ComplexNum, inumber2: ComplexNum) -> ComplexNum:
    """Returns the quotient of two complex numbers"""
    pass


@complex_func(cmath.exp, result2str=True)
def IMEXP(inumber: ComplexNum) -> ComplexNum:
    """Returns the exponential of a complex number"""
    pass


@complex_func(cmath.log, result2str=True)
def IMLN(inumber: ComplexNum) -> ComplexNum:
    """Returns the natural logarithm of a complex number"""
    pass


@complex_func(cmath.log10, result2str=True)
def IMLOG10(inumber: ComplexNum) -> ComplexNum:
    """Returns the base-10 logarithm of a complex number"""
    pass


@complex_func(lambda x: cmath.log(x, 2), result2str=True)
def IMLOG2(inumber: ComplexNum) -> ComplexNum:
    """Returns the base-2 logarithm of a complex number"""
    pass


@complex_func(pow, result2str=True, complex_args=[0])
def IMPOWER(inumber: ComplexNum, number: Numeric) -> ComplexNum:
    """Returns a complex number raised to an integer power"""
    pass


@complex_func(lambda *args: reduce(mul, args), result2str=True, range_args=True)
def IMPRODUCT(
    inumber1: ComplexNum | CellRange, *inumbers: tuple[ComplexNum | CellRange]
) -> ComplexNum:
    """Returns the product of from 2 to 255 complex numbers"""
    pass


@complex_func(lambda x: x.real)
def IMREAL(inumber: ComplexNum) -> Numeric:
    """Returns the real coefficient of a complex number"""
    pass


@complex_func(lambda x: 1 / cmath.cos(x), result2str=True)
def IMSEC(inumber: ComplexNum) -> ComplexNum:
    """Returns the secant of a complex number"""
    pass


@complex_func(lambda x: 1 / cmath.cosh(x), result2str=True)
def IMSECH(inumber: ComplexNum) -> ComplexNum:
    """Returns the hyperbolic secant of a complex number"""
    pass


@complex_func(cmath.sin, result2str=True)
def IMSIN(inumber: ComplexNum) -> ComplexNum:
    """Returns the sine of a complex number"""
    pass


@complex_func(cmath.sinh, result2str=True)
def IMSINH(inumber: ComplexNum) -> ComplexNum:
    """Returns the hyperbolic sine of a complex number"""
    pass


@complex_func(cmath.sqrt, result2str=True)
def IMSQRT(inumber: ComplexNum) -> ComplexNum:
    """Returns the square root of a complex number"""
    pass


@complex_func(sub, result2str=True)
def IMSUB(inumber1: ComplexNum, inumber2: ComplexNum) -> ComplexNum:
    """Returns the difference between two complex numbers"""
    pass


@complex_func(lambda *args: reduce(add, args), result2str=True, range_args=True)
def IMSUM(
    inumber1: ComplexNum | CellRange, *inumbers: tuple[ComplexNum | CellRange]
) -> ComplexNum:
    """Returns the sum of complex numbers"""
    pass


@complex_func(cmath.tan, result2str=True)
def IMTAN(inumber: ComplexNum) -> ComplexNum:
    """Returns the tangent of a complex number"""
    pass


@convert_base(8, 2)
def OCT2BIN(number: str, places: int | None = None) -> str:
    """Converts an octal number to binary"""
    pass


def OCT2DEC(number: str) -> Numeric:
    """Converts an octal number to decimal"""
    return to_decimal(number, 8, _CHAR_LIMIT)


@convert_base(8, 16)
def OCT2HEX(number: str, places: int | None = None) -> str:
    """Converts an octal number to hexadecimal"""
    pass
