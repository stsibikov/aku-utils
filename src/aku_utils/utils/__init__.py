def process_cols_input(value: list[str] | str) -> list[str]:
    """
    Args
    ---
        value:
            if list, just returns it.

            if string with tabs in it, splits the string
            by tabs.

            if string with no tabs, returns it as a list
            with one value.
    """
    if isinstance(value, list):
        return value

    if isinstance(value, str) and "\t" in value:
        value = value.split("\t")
    elif isinstance(value, str):
        value = [value]
    return value
