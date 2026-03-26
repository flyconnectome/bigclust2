import colorsys

import polars as pl

from pathlib import Path
from urllib.parse import urlparse, urlunparse, urljoin


def is_url(string: str) -> bool:
    """Check if a string is a URL.

    Args:
        string: The string to check

    Returns:
        True if the string is a URL, False otherwise
    """
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def is_relative(path: str | Path) -> bool:
    """Check if a path is relative.

    Args:
        path: The path to check (string or Path object)

    Returns:
        True if the path is relative, False if absolute
    """
    return not Path(path).is_absolute()


def is_list_of_ids(string: str) -> bool:
    """Check if a string represents a list of IDs (comma-separated values).

    Args:
        string: The string to check
    Returns:
        True if the string is a list of IDs, False otherwise
    """
    if not isinstance(string, str) or not string.strip():
        return False

    parts = [part.strip() for part in string.split(",")]
    for part in parts:
        if not part:
            return False
        if not part.isdigit():
            return False
    return True


def string_to_polars_filter(filter_expr: str) -> pl.Expr:
    """Convert a string filter expression to a Polars expression.

    Args:
        filter_expr: The filter expression as a string (e.g. 'column_a > 5 & column_b == "test"')

    Returns:
        A Polars expression representing the filter
    """
    if not isinstance(filter_expr, str) or not filter_expr.strip():
        raise ValueError("filter_expr must be a non-empty string")

    s = filter_expr.strip()

    def is_ident_start(ch: str) -> bool:
        return ch.isalpha() or ch == "_"

    def is_ident_part(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    class Tok:
        def __init__(self, t: str, v):
            self.t = t
            self.v = v

        def __repr__(self):
            return f"Tok({self.t},{self.v})"

    i = 0
    n = len(s)
    tokens = []

    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(Tok("LPAREN" if ch == "(" else "RPAREN", ch))
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            i += 1
            buf = []
            while i < n:
                c = s[i]
                if c == "\\" and i + 1 < n:
                    buf.append(s[i + 1])
                    i += 2
                    continue
                if c == quote:
                    i += 1
                    break
                buf.append(c)
                i += 1
            tokens.append(Tok("STRING", "".join(buf)))
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and s[i + 1].isdigit()):
            j = i
            has_dot = False
            while j < n and (s[j].isdigit() or s[j] == "."):
                if s[j] == ".":
                    if has_dot:
                        break
                    has_dot = True
                j += 1
            num_str = s[i:j]
            val = float(num_str) if has_dot else int(num_str)
            tokens.append(Tok("NUMBER", val))
            i = j
            continue
        if is_ident_start(ch):
            j = i + 1
            while j < n and is_ident_part(s[j]):
                j += 1
            ident = s[i:j]
            low = ident.lower()
            if low == "and":
                tokens.append(Tok("OP", "&"))
                i = j
            elif low == "or":
                tokens.append(Tok("OP", "|"))
                i = j
            elif low == "in":
                # Handle "in" operator with list: in (val1, val2, ...) or in [val1, val2, ...]
                # Skip whitespace after "in"
                k = j
                while k < n and s[k].isspace():
                    k += 1
                if k < n and s[k] in "([":
                    # Parse the list
                    closing = ")" if s[k] == "(" else "]"
                    k += 1
                    list_items = []
                    while k < n:
                        # Skip whitespace
                        while k < n and s[k].isspace():
                            k += 1
                        if k >= n:
                            raise ValueError(
                                f"Unclosed list in 'in' operator: expected '{closing}' before end of string"
                            )
                        if s[k] == closing:
                            k += 1
                            break
                        # Parse a value (string, number)
                        if s[k] in "'\"":
                            quote = s[k]
                            k += 1
                            buf = []
                            found_closing_quote = False
                            while k < n:
                                c = s[k]
                                if c == "\\" and k + 1 < n:
                                    buf.append(s[k + 1])
                                    k += 2
                                    continue
                                if c == quote:
                                    k += 1
                                    found_closing_quote = True
                                    break
                                buf.append(c)
                                k += 1
                            if not found_closing_quote:
                                raise ValueError(
                                    f"Unclosed string in list at pos {k}: expected '{quote}'"
                                )
                            list_items.append("".join(buf))
                        elif s[k].isdigit() or (
                            s[k] == "." and k + 1 < n and s[k + 1].isdigit()
                        ):
                            start = k
                            has_dot = False
                            while k < n and (s[k].isdigit() or s[k] == "."):
                                if s[k] == ".":
                                    if has_dot:
                                        break
                                    has_dot = True
                                k += 1
                            num_str = s[start:k]
                            val = float(num_str) if has_dot else int(num_str)
                            list_items.append(val)
                        else:
                            raise ValueError(
                                f"Unexpected token in list at pos {k}: {s[k]}"
                            )
                        # Skip whitespace and comma
                        while k < n and s[k].isspace():
                            k += 1
                        if k < n and s[k] == ",":
                            k += 1
                    tokens.append(Tok("IN", "in"))
                    tokens.append(Tok("LIST", tuple(list_items)))
                    i = k
                else:
                    raise ValueError(
                        f"'in' operator must be followed by a list in parentheses or brackets at pos {j}"
                    )
            else:
                tokens.append(Tok("IDENT", ident))
                i = j
            continue
        # operators
        two = s[i : i + 2]
        if two in ("==", "!=", ">=", "<="):
            tokens.append(Tok("OP", two))
            i += 2
            continue
        if ch in (">", "<", "&", "|"):
            tokens.append(Tok("OP", ch))
            i += 1
            continue
        raise ValueError(f"Unexpected character in filter expression at pos {i}: {ch}")

    prec = {
        "==": 3,
        "!=": 3,
        ">": 3,
        ">=": 3,
        "<": 3,
        "<=": 3,
        "in": 3,
        "&": 2,
        "|": 1,
    }

    output = []
    op_stack = []

    for tok in tokens:
        if tok.t in ("IDENT", "NUMBER", "STRING"):
            output.append(tok)
        elif tok.t == "LIST":
            output.append(tok)
        elif tok.t == "IN":
            while op_stack:
                top = op_stack[-1]
                if top.t == "OP" and prec[top.v] >= prec["in"]:
                    output.append(op_stack.pop())
                elif top.t == "IN":
                    output.append(op_stack.pop())
                else:
                    break
            op_stack.append(tok)
        elif tok.t == "OP":
            while op_stack:
                top = op_stack[-1]
                if top.t == "OP" and prec[top.v] >= prec[tok.v]:
                    output.append(op_stack.pop())
                elif top.t == "IN" and prec["in"] >= prec[tok.v]:
                    output.append(op_stack.pop())
                else:
                    break
            op_stack.append(tok)
        elif tok.t == "LPAREN":
            op_stack.append(tok)
        elif tok.t == "RPAREN":
            while op_stack and op_stack[-1].t != "LPAREN":
                output.append(op_stack.pop())
            if not op_stack:
                raise ValueError("Mismatched parentheses in filter expression")
            op_stack.pop()
    while op_stack:
        t = op_stack.pop()
        if t.t in ("LPAREN", "RPAREN"):
            raise ValueError("Mismatched parentheses in filter expression")
        output.append(t)

    def to_expr(tok: Tok):
        if tok.t == "IDENT":
            return pl.col(tok.v)
        if tok.t == "NUMBER" or tok.t == "STRING":
            return pl.lit(tok.v)
        raise ValueError(f"Invalid token to convert: {tok}")

    stack = []
    for tok in output:
        if tok.t in ("IDENT", "NUMBER", "STRING"):
            stack.append(to_expr(tok))
        elif tok.t == "LIST":
            stack.append(tok.v)
        elif tok.t == "IN":
            if len(stack) < 2:
                raise ValueError("Invalid expression: insufficient operands for 'in'")
            list_val = stack.pop()
            col_expr = stack.pop()
            if not isinstance(list_val, tuple):
                raise ValueError("Right operand of 'in' must be a list")
            stack.append(col_expr.is_in(list_val))
        elif tok.t == "OP":
            if len(stack) < 2:
                raise ValueError("Invalid expression: insufficient operands")
            rhs = stack.pop()
            lhs = stack.pop()
            if tok.v == "==":
                stack.append(lhs == rhs)
            elif tok.v == "!=":
                stack.append(lhs != rhs)
            elif tok.v == ">":
                stack.append(lhs > rhs)
            elif tok.v == ">=":
                stack.append(lhs >= rhs)
            elif tok.v == "<":
                stack.append(lhs < rhs)
            elif tok.v == "<=":
                stack.append(lhs <= rhs)
            elif tok.v == "&":
                stack.append(lhs & rhs)
            elif tok.v == "|":
                stack.append(lhs | rhs)
            else:
                raise ValueError(f"Unsupported operator: {tok.v}")
        else:
            raise ValueError(f"Unexpected token in RPN: {tok}")

    if len(stack) != 1:
        raise ValueError("Invalid filter expression")
    return stack[0]


class Url:
    """Class representing a URL.

    This is analogous to pathlib.Path but for URLs and
    with limited functionality.
    """

    def __init__(self, url: str):
        if isinstance(url, Url):
            self.url = url.url
        elif not isinstance(url, str):
            raise TypeError("Url must be initialized with a string or Url object")

        self.url = url

    def __str__(self):
        return self.url

    def __repr__(self):
        return f"Url({self.url})"

    def __div__(self, other) -> "Url":
        return self.joinpath(other)

    def __truediv__(self, other) -> "Url":
        return self.joinpath(other)

    def joinpath(self, *args) -> "Url":
        new_url = self.url
        for arg in args:
            if not isinstance(arg, str):
                arg = str(arg)

            if not new_url.endswith("/"):
                new_url += "/"
            new_url = urljoin(new_url, arg)
        return Url(new_url)

    @property
    def parent(self) -> "Url":
        parsed = urlparse(self.url)
        path_parts = parsed.path.rstrip("/").split("/")
        parent_path = "/".join(path_parts[:-1])
        new_parsed = parsed._replace(path=parent_path)
        return Url(urlunparse(new_parsed))

    @property
    def name(self) -> str:
        parsed = urlparse(self.url)
        path_parts = parsed.path.rstrip("/").split("/")
        return path_parts[-1] if path_parts else ""

    @property
    def suffix(self) -> str:
        name = self.name
        if "." in name:
            return name[name.rfind(".") :]
        return ""

    @property
    def stem(self) -> str:
        name = self.name
        if "." in name:
            return name[: name.rfind(".")]
        return name


def hash_function(state, value):
    """This is a modified murmur hash.
    """
    k1 = 0xCC9E2D51
    k2 = 0x1B873593
    state = state & 0xFFFFFFFF
    value = (value * k1) & 0xFFFFFFFF
    value = ((value << 15) | value >> 17) & 0xFFFFFFFF
    value = (value * k2) & 0xFFFFFFFF
    state = (state ^ value) & 0xFFFFFFFF
    state = ((state << 13) | state >> 19) & 0xFFFFFFFF
    state = ((state * 5) + 0xE6546B64) & 0xFFFFFFFF
    return state


def rgb_from_segment_id(color_seed, segment_id):
    """Return the RGBA for a segment given a color seed and the segment ID."""
    segment_id = int(segment_id)  # necessary since segment_id is 64 bit originally
    result = hash_function(state=color_seed, value=segment_id)
    newvalue = segment_id >> 32
    result2 = hash_function(state=result, value=newvalue)
    c0 = (result2 & 0xFF) / 255.0
    c1 = ((result2 >> 8) & 0xFF) / 255.0
    h = c0
    s = 0.5 + 0.5 * c1
    v = 1.0
    return tuple([v * 255 for v in colorsys.hsv_to_rgb(h, s, v)])