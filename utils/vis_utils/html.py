from typing import Any

import fasthtml.common as ft


def create_page(elements: list[ft.FT]) -> ft.FT:
    return ft.Html(
        ft.Head(
            ft.picolink,
            ft.Link(
                rel="stylesheet",
                href="https://cdnjs.cloudflare.com/ajax/libs/flexboxgrid/6.3.1/flexboxgrid.min.css",
                type="text/css",
            ),
        ),
        ft.Body(ft.Container(*elements)),
    )


def save_html(elements: list[ft.FT], filepath: str):
    with open(filepath, "w") as file:
        file.write(ft.to_xml(create_page(elements)))


def create_table(columns: list[str], rows: list[dict[str, Any]]):
    header = ft.Tr(*[ft.Th(col) for col in columns])
    body = []
    for row in rows:
        body.append(ft.Tr(*[ft.Td(row.get(col, "")) for col in columns]))
    return ft.Table(ft.Thead(header), ft.Tbody(*body))
