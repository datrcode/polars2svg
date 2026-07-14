class Polars2SVGError(Exception):
    """Base class for all errors raised by polars2svg library code."""

class InvalidSpecError(Polars2SVGError):
    """Raised for invalid count=/color=/field specs or malformed parameter values."""

class DataError(Polars2SVGError):
    """Raised for empty/missing columns, bad dtypes, or otherwise malformed input data."""
