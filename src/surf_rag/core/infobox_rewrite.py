from bs4 import BeautifulSoup
from surf_rag.core.corpus_schemas import Block


def is_infobox_table(table) -> bool:
    """Check if a given table is an infobox."""
    if "infobox" in table.attrs.get("class", []):
        return True
    return False


def linearize_infobox_table(table, page_title: str = "") -> list[Block]:
    """Linearize a given infobox table into a string of text."""

    infobox_text = ""

    # Get subject
    title = table.find("div", class_="fn")
    subject = ""
    if title:
        subject = title.get_text()
    else:
        subject = page_title

    # Iterate through rows
    rows = table.find_all("tr")
    for row in rows:
        label = row.find("th", class_="infobox-label")
        if label:
            # Text
            value_cell = row.find("td", class_="infobox-data")
            values = value_cell.find_all("span", recursive=False)
            for value in values:
                if value.get_text() != "":
                    infobox_text += (
                        f"{subject} -- {label.get_text()} --> {value.get_text()}\n"
                    )

            # Links
            links = row.find_all("a")
            if links and links[0].find_parent("div", class_="birthplace"):
                birthplace = ""
                for link in links:
                    if link.get_text() != "":
                        birthplace += link.get_text() + ", "
                infobox_text += (
                    f"{subject} -- {label.get_text()} --> {birthplace[:-2]}\n"
                )
            else:
                for link in links:
                    if link.get_text() != "":
                        infobox_text += (
                            f"{subject} -- {label.get_text()} --> {link.get_text()}\n"
                        )

            # Handle nested lists
            list_elements = row.find_all("li")
            for list_element in list_elements:
                if list_element.get_text() != "":
                    infobox_text += f"{subject} -- {label.get_text()} --> {list_element.get_text()}\n"

    return infobox_text.strip()


# Testing
if __name__ == "__main__":
    with open("temp/jy.html", "r") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="infobox")
    if is_infobox_table(table):
        print(linearize_infobox_table(table, "James Young"))
    else:
        print("Not an infobox table")
